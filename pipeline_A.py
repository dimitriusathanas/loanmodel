"""
pipeline_A.py — Deliverable A: single importable module.

Merges features.py, model.py, policy.py, bootstrap interval step, and
submission-writing into one self-contained, side-effect-free module.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B-HANDOFF API  (import pipeline_A and call these directly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    build_features(df) -> pd.DataFrame
        Full feature matrix for any raw split (train / val / test).
        B reuses this directly for its own cohort-level model.

    get_calibrated_pd(df) -> pd.Series
        Calibrated PD for every row in df, indexed by applicant_id.
        Lazily trains and caches the production model on first call.

    get_approved_set() -> np.ndarray[str]
        Applicant IDs where decision == 1 (B's approved population A_w).
        Reads from the locked submission_A_decisions.csv.

    assign_cohort_week(df) -> pd.Series
        Maps application_timestamp to cohort_week 1-13 via
        dataset/cohort_week_definitions.csv.  Rows outside the 13-week
        window get NaN.  Pass val_raw or test_raw directly.

    get_train_timing() -> np.ndarray[float]
        days_to_default for all train defaulters.
        B's empirical distribution for loan-age → default timing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Running directly reproduces the locked Deliverable A submission:
    python pipeline_A.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parent
SUBMISSION_DIR = BASE / "submission"

# ══════════════════════════════════════════════════════════════════════════════
# ── FEATURES (shared with B) ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

OUTCOME_COLS: list[str] = [
    "default_flag", "days_to_default", "days_to_full_repayment",
    "repayment_status", "final_recovered_amount", "observation_status",
]

CATEGORICAL_FEATURES: list[str] = ["sector", "overdrawn_state"]

FEATURE_COLS: list[str] = [
    "daily_draw", "stress_cash", "overdrawn_state", "stress_rev",
    "aggregate_credit_utilization", "invoice_payment_delinquency_rate",
    "owner_personal_credit_band", "existing_debt_obligations",
    "prior_underwriter_score", "was_prior_declined", "prior_approved_amount",
    "has_prior_external_decline", "has_prior_inquiry_elsewhere",
    "multi_lender_inquiry_count_30d", "repeat_application_count",
    "overdraft_count", "sector",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from any raw split DataFrame.

    NaN values are left as-is for LightGBM null routing (bank-feed rows only).
    Never touches outcome columns.  Safe to call on train, val, or test.
    """
    out = pd.DataFrame(index=df.index)

    out["daily_draw"]  = df["requested_amount"] * 1.0575 / 60
    out["stress_cash"] = df["observed_cash_balance_p10"] / out["daily_draw"]

    has_feed = df["has_linked_bank_feed"] == True
    cash_ok  = has_feed & (df["observed_cash_balance_p10"] >= 0)
    cash_ov  = has_feed & (df["observed_cash_balance_p10"] < 0)
    state = np.zeros(len(df), dtype=np.int32)
    state[cash_ok.values] = 1
    state[cash_ov.values] = 2
    out["overdrawn_state"] = state

    out["stress_rev"]                        = (df["observed_monthly_revenue_avg_3mo"] / 30) / out["daily_draw"]
    out["aggregate_credit_utilization"]      = df["aggregate_credit_utilization"]
    out["invoice_payment_delinquency_rate"]  = df["invoice_payment_delinquency_rate"]
    out["owner_personal_credit_band"]        = df["owner_personal_credit_band"]
    out["existing_debt_obligations"]         = df["existing_debt_obligations"]
    out["prior_underwriter_score"]           = df["prior_underwriter_score"]
    out["was_prior_declined"]                = df["prior_approved_amount"].isna().astype(np.int8)
    out["prior_approved_amount"]             = df["prior_approved_amount"].fillna(0.0)
    out["has_prior_external_decline"]        = df["days_since_last_external_decline"].notna().astype(np.int8)
    out["has_prior_inquiry_elsewhere"]       = df["days_since_last_inquiry_elsewhere"].notna().astype(np.int8)
    out["multi_lender_inquiry_count_30d"]    = df["multi_lender_inquiry_count_30d"]
    out["repeat_application_count"]          = df["repeat_application_count"]
    out["overdraft_count"]                   = df["observed_overdraft_count_3mo"]
    out["sector"]                            = df["sector"].astype(np.int32)

    leaked = [c for c in OUTCOME_COLS if c in out.columns]
    assert not leaked, f"LEAKAGE: {leaked}"
    return out[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# ── MODEL ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_MONOTONE_CONSTRAINTS: list[int] = [
     0,   # daily_draw
    -1,   # stress_cash:                   higher cushion → lower PD
     0,   # overdrawn_state (categorical)
    -1,   # stress_rev:                    higher headroom → lower PD
     1,   # aggregate_credit_utilization:  higher → more risk
     1,   # invoice_payment_delinquency_rate
    -1,   # owner_personal_credit_band:    higher band → lower PD
     1,   # existing_debt_obligations
    -1,   # prior_underwriter_score:       higher → lower PD
     0,   # was_prior_declined
     0,   # prior_approved_amount
     0,   # has_prior_external_decline
     0,   # has_prior_inquiry_elsewhere
     1,   # multi_lender_inquiry_count_30d
     0,   # repeat_application_count
     1,   # overdraft_count
     0,   # sector (categorical)
]

_LGB_PARAMS: dict = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    monotone_constraints=_MONOTONE_CONSTRAINTS,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


def _train_model(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
) -> tuple[lgb.LGBMClassifier, IsotonicRegression]:
    """
    Train LightGBM on labeled train rows; calibrate isotonic on labeled val rows.
    Returns (model, calibrator).
    """
    labeled = train_raw[train_raw["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").reset_index(drop=True)
    split   = int(len(ordered) * 0.80)
    tr_fit  = ordered.iloc[:split]
    tr_es   = ordered.iloc[split:]

    model = lgb.LGBMClassifier(**_LGB_PARAMS)
    model.fit(
        build_features(tr_fit), tr_fit["default_flag"].values.astype(int),
        eval_set=[(build_features(tr_es), tr_es["default_flag"].values.astype(int))],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    labeled_val = val_raw[val_raw["default_flag"].notna()].copy()
    raw_cal     = model.predict_proba(build_features(labeled_val))[:, 1]
    calibrator  = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, labeled_val["default_flag"].values.astype(int))
    return model, calibrator


def _predict_calibrated(
    model: lgb.LGBMClassifier,
    calibrator: IsotonicRegression,
    df: pd.DataFrame,
) -> np.ndarray:
    """Calibrated PD ∈ [0,1] for every row in df."""
    return np.clip(calibrator.transform(model.predict_proba(build_features(df))[:, 1]), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ── POLICY ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Locked P&L-optimal decision threshold from empirical sweep on labeled train.
# Analytic break-even (timing-aware): p* = 0.3686; empirical sweep finds 0.3400.
P_STAR: float = 0.3400


def loan_npv_vec(
    R: np.ndarray,
    default_flag: np.ndarray,
    days_to_default: np.ndarray,
    recovery: np.ndarray,
) -> np.ndarray:
    """
    Vectorized per-loan realized NPV (timing-aware brief formula).
      Repaid:  NPV = 0.0875 * R
      Default: NPV = 0.03*R + D*(t*-1) + recovery - R   where D = R*1.0575/60
    """
    R    = np.asarray(R, dtype=float)
    flag = np.asarray(default_flag, dtype=int)
    t    = np.where(np.isnan(np.asarray(days_to_default, dtype=float)), 0.0,
                    np.asarray(days_to_default, dtype=float))
    rec  = np.where(np.isnan(np.asarray(recovery, dtype=float)), 0.0,
                    np.asarray(recovery, dtype=float))
    D           = R * 1.0575 / 60
    npv_good    = 0.0875 * R
    npv_default = 0.03 * R + D * (t - 1) + rec - R
    return np.where(flag == 0, npv_good, npv_default)


def _make_decisions(calibrated_pd: np.ndarray, p_star: float = P_STAR) -> np.ndarray:
    return (np.asarray(calibrated_pd) < p_star).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# ── BOOTSTRAP INTERVALS ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_N_BOOTSTRAP  = 50
_NO_FEED_HW   = 0.05   # additive half-width per side: no bank-feed rows
_LOW_SCORE_HW = 0.03   # additive half-width per side: reject-inference region


def _build_intervals(
    predicted_pd: np.ndarray,
    score_df: pd.DataFrame,
    labeled_train: pd.DataFrame,
    labeled_val: pd.DataFrame,
    y_val: np.ndarray,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N_BOOTSTRAP calibrated bootstrap iterations; return (lower, upper) for
    all len(score_df) applicants.  Symmetric half-width centred at predicted_pd.
    """
    boot_params = {**_LGB_PARAMS, "n_estimators": 200}
    y_labeled   = labeled_train["default_flag"].values.astype(int)
    n_labeled   = len(labeled_train)

    X_labeled = build_features(labeled_train)
    X_val_f   = build_features(labeled_val)
    X_score   = build_features(score_df)

    boot_preds = np.empty((_N_BOOTSTRAP, len(score_df)), dtype=np.float32)
    t0 = time.time()

    for b in range(_N_BOOTSTRAP):
        rng = np.random.default_rng(seed=42 + b)
        idx = rng.integers(0, n_labeled, size=n_labeled)

        m = lgb.LGBMClassifier(**boot_params)
        m.fit(X_labeled.iloc[idx], y_labeled[idx],
              categorical_feature=CATEGORICAL_FEATURES)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(X_val_f)[:, 1], y_val)

        boot_preds[b] = np.clip(
            iso.transform(m.predict_proba(X_score)[:, 1]), 0.0, 1.0
        ).astype(np.float32)

        if not silent and ((b + 1) % 10 == 0 or b == 0):
            elapsed = time.time() - t0
            eta     = elapsed / (b + 1) * (_N_BOOTSTRAP - b - 1)
            print(f"  [{b+1:2d}/{_N_BOOTSTRAP}]  {elapsed:4.0f}s  eta={eta:3.0f}s")

    pct05 = np.percentile(boot_preds, 5,  axis=0).astype(float)
    pct95 = np.percentile(boot_preds, 95, axis=0).astype(float)
    hw    = (pct95 - pct05) / 2.0
    hw    = np.maximum(hw, np.abs(predicted_pd - (pct05 + pct95) / 2.0))

    # Region-aware widening
    is_no_feed   = (~score_df["has_linked_bank_feed"]).values.astype(float)
    pus_vals     = score_df["prior_underwriter_score"].values.astype(float)
    pus_thr      = np.nanpercentile(pus_vals, 20)
    is_low_score = (pus_vals <= pus_thr).astype(float)
    total_hw     = hw + _NO_FEED_HW * is_no_feed + _LOW_SCORE_HW * is_low_score

    lower = np.minimum(np.clip(predicted_pd - total_hw, 0.0, 1.0), predicted_pd)
    upper = np.maximum(np.clip(predicted_pd + total_hw, 0.0, 1.0), predicted_pd)
    return lower, upper


# ══════════════════════════════════════════════════════════════════════════════
# ── B-HANDOFF API ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_CACHE: dict = {"model": None, "calibrator": None}


def _ensure_model() -> None:
    """Lazy-initialise production model + calibrator (trains once, then caches)."""
    if _MODEL_CACHE["model"] is not None:
        return
    train_raw = pd.read_csv(BASE / "dataset/train.csv")
    val_raw   = pd.read_csv(BASE / "dataset/validation.csv")
    _MODEL_CACHE["model"], _MODEL_CACHE["calibrator"] = _train_model(train_raw, val_raw)


def get_calibrated_pd(df: pd.DataFrame) -> pd.Series:
    """
    Return calibrated PD for every row in df, indexed by applicant_id.

    Trains and caches the production model on the first call (~15 s);
    subsequent calls are fast (predict only).
    """
    _ensure_model()
    pd_arr = _predict_calibrated(_MODEL_CACHE["model"], _MODEL_CACHE["calibrator"], df)
    return pd.Series(pd_arr, index=df["applicant_id"].values, name="calibrated_pd")


def get_approved_set() -> np.ndarray:
    """
    Return applicant_ids (str array) where decision == 1.

    This is B's approved population A_w — the set of approved applicants
    whose default trajectory B must forecast.  Reads from the locked
    submission_A_decisions.csv produced by pipeline_A.
    """
    sub = pd.read_csv(SUBMISSION_DIR / "submission_A_decisions.csv")
    return sub.loc[sub["decision"] == 1, "applicant_id"].values


def assign_cohort_week(df: pd.DataFrame) -> pd.Series:
    """
    Map each row's application_timestamp to cohort_week 1-13.

    Uses dataset/cohort_week_definitions.csv.  Rows whose timestamp falls
    outside the 13-week window get NaN (these should not appear in val/test).

    Parameters
    ----------
    df : raw DataFrame with an application_timestamp column

    Returns
    -------
    pd.Series of float (1.0 .. 13.0 or NaN), same index as df
    """
    cw_def = pd.read_csv(BASE / "dataset/cohort_week_definitions.csv")
    cw_def["start_date"] = pd.to_datetime(cw_def["start_date"])
    cw_def["end_date"]   = pd.to_datetime(cw_def["end_date"])

    dates  = pd.to_datetime(df["application_timestamp"].astype(str).str[:10])
    result = pd.Series(np.nan, index=df.index, name="cohort_week", dtype=float)
    for _, row in cw_def.iterrows():
        mask = (dates >= row["start_date"]) & (dates <= row["end_date"])
        result.loc[mask] = float(row["cohort_week"])
    return result


def get_train_timing() -> np.ndarray:
    """
    Return days_to_default for all labeled train defaulters.

    B uses this as the empirical distribution of when defaults occur within
    a loan term — the primary input for building the 13×13 trajectory grid.
    """
    train_raw = pd.read_csv(BASE / "dataset/train.csv")
    defaulters = train_raw[
        (train_raw["default_flag"] == 1) & train_raw["days_to_default"].notna()
    ]
    return defaulters["days_to_default"].values.astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# ── SUBMISSION HELPERS ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _write_stub_b(out_dir: Path) -> None:
    template = pd.read_csv(BASE / "dataset/submission_B_template.csv")
    template["cumulative_default_rate"] = 0.0
    template["cdr_lower_90"]            = 0.0
    template["cdr_upper_90"]            = 0.0
    template.to_csv(out_dir / "submission_B_trajectory.csv", index=False)


def _write_stub_c(out_dir: Path) -> None:
    queries = pd.read_csv(BASE / "dataset/intervention_queries.csv")
    pd.DataFrame({
        "query_id":        queries["query_id"],
        "predicted_pd_cf": 0.5,
        "pd_cf_lower_90":  0.42,
        "pd_cf_upper_90":  0.58,
    }).to_csv(out_dir / "submission_C_counterfactuals.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN (reproduces the locked A submission) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    SUBMISSION_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    train_raw = pd.read_csv(BASE / "dataset/train.csv")
    val_raw   = pd.read_csv(BASE / "dataset/validation.csv")
    test_raw  = pd.read_csv(BASE / "dataset/test.csv")

    labeled_train = train_raw[train_raw["default_flag"].notna()].copy()
    labeled_val   = val_raw[val_raw["default_flag"].notna()].copy()
    y_val         = labeled_val["default_flag"].values.astype(int)

    # 13,306 applicants in submission order
    score_df = pd.concat([val_raw, test_raw], ignore_index=True)
    assert len(score_df) == 13_306

    # ── Train production model ────────────────────────────────────────────────
    print("\nTraining production model...")
    model, calibrator = _train_model(train_raw, val_raw)
    _MODEL_CACHE["model"]      = model        # populate cache for API callers
    _MODEL_CACHE["calibrator"] = calibrator

    # ── Score all 13,306 applicants ───────────────────────────────────────────
    print("Scoring 13,306 applicants...")
    calibrated_pd = _predict_calibrated(model, calibrator, score_df)
    decisions     = _make_decisions(calibrated_pd, P_STAR)
    n_approve     = int(decisions.sum())
    print(f"  Approvals: {n_approve:,} / {len(decisions):,}  "
          f"({n_approve/len(decisions)*100:.1f}%)  threshold={P_STAR}")

    # ── Bootstrap intervals ───────────────────────────────────────────────────
    print(f"\nBuilding bootstrap intervals (N={_N_BOOTSTRAP})...")
    lower, upper = _build_intervals(
        calibrated_pd, score_df, labeled_train, labeled_val, y_val
    )

    feed_mask  = score_df["has_linked_bank_feed"].values == True
    width      = upper - lower
    print(f"  mean width — overall={width.mean():.4f}  "
          f"feed={width[feed_mask].mean():.4f}  "
          f"no-feed={width[~feed_mask].mean():.4f}")

    # ── Write Deliverable A ───────────────────────────────────────────────────
    sub = pd.DataFrame({
        "applicant_id": score_df["applicant_id"].values,
        "decision":     decisions,
        "predicted_pd": calibrated_pd,
        "pd_lower_90":  lower,
        "pd_upper_90":  upper,
    })
    out_a = SUBMISSION_DIR / "submission_A_decisions.csv"
    sub.to_csv(out_a, index=False)
    print(f"\nWrote {out_a}  ({len(sub):,} rows)")

    _write_stub_b(SUBMISSION_DIR)
    _write_stub_c(SUBMISSION_DIR)

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    result = subprocess.run(
        [sys.executable, str(BASE / "validate_submission.py"), str(SUBMISSION_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

    return 0 if "PASS" in result.stdout else 1


if __name__ == "__main__":
    raise SystemExit(main())
