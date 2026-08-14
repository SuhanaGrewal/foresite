"""
Train Predictor
 
Trains a cache-reuse predictor from a features CSV produced by
feature_extraction.py.
 
Two models are trained:
- Logistic regression: baseline, coefficients printed for inspection
- XGBoost: primary model
 
Evaluated two ways:
- a single trace-level train/test split (kept for the printed baseline
  coefficients and as the split the persisted model is trained on)
- GroupKFold cross-validation (group = trace_id, so no trace's rows everx
  span both sides of a fold), reporting AUC-ROC/PR-AUC mean +/- standard
  deviation across folds -- a far more reliable read on generalization
  than any single split, now that there's enough trace data (100+
  traces) for folds to be meaningful.

Also reports a calibration check: each row's out-of-fold predicted
probability (from the cross-validation above -- every row gets exactly
one prediction, made by a model that never trained on it) is bucketed
into 0.0-0.1, 0.1-0.2, ... 0.9-1.0, and each bucket's actual observed
will_be_reused rate is compared against the model's own mean predicted
probability in that bucket. A well-calibrated "70% reuse probability"
prediction should see roughly 70% of that bucket's rows actually get
reused -- this check is what tells you whether that's true or whether
the model's probabilities are just a good *ranking* (which AUC-ROC/PR-AUC
already confirm) without being meaningfully interpretable as real
probabilities.

Optional --extra-input lets a second features CSV (e.g. features_sweep.csv)
be folded into training, up-weighted via --extra-weight-fraction so a
small extra source isn't drowned out by raw row-count alone (e.g. sweep
traces were only ~6% of combined rows unweighted). This exists because
the trained model, evaluated against a genuinely different access
pattern (a "sweep": many concurrent one-off touches, see
run_sweep_eviction_benchmark.py), performed no better than LRU there --
its dominant learned feature (event_type_model_call) is a strong signal
in the general 2-5-subtask task shape but doesn't transfer, since the
model had never seen a sweep-shaped trace during training. Training
sample weights are the fix; evaluation (AUC/PR-AUC/calibration) stays
unweighted throughout, so reported numbers reflect real observed rates,
not an artificially rebalanced view.
"""

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# unscaled features + lbfgs occasionally hits max_iter without converging;
# benign here (this is a baseline for coefficient inspection, not the
# primary model), but cross-validation now fits it 5x more often than a
# single split did, so it's worth silencing rather than burying the actual
# results under repeated identical warnings.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from xgboost import XGBClassifier

    PRIMARY_MODEL_NAME = "xgboost"
except ImportError:
    from lightgbm import LGBMClassifier as XGBClassifier

    PRIMARY_MODEL_NAME = "lightgbm"

RANDOM_SEED = 42
N_CV_SPLITS = 5
TARGET_COLUMN = "will_be_reused"
ID_COLUMNS = ["item_id", "timestamp", "agent_id", "parent_id"]
WEIGHT_COLUMN = "_sample_weight"

NON_FEATURE_COLUMNS = ID_COLUMNS + ["event_type"]

MISSING_VALUE_SENTINEL = -1

# default target: extra-source rows make up roughly this fraction of total
# TRAINING WEIGHT (not row count) once combined with the main dataset.
# chosen as a middle ground -- high enough that a small extra source (e.g.
# 3 sweep traces, ~6% of combined rows unweighted) has real influence on
# what the model learns, low enough that the model isn't overfit to
# memorizing a handful of traces' idiosyncrasies at full parity (50%).
DEFAULT_EXTRA_WEIGHT_FRACTION = 0.25


def clean_feature_values(df):
    """
    shared cleaning step: bool -> int, and NaN -> MISSING_VALUE_SENTINEL (the
    same sentinel steps_since_last_seen/seconds_since_last_seen already use
    for "not applicable"). exposed (not module-private) because
    run_eviction_benchmark.py's feature_lookup needs this exact same
    transform applied to features.csv rows before scoring them with the
    trained model -- model.joblib's model was fit on data cleaned this way,
    so anything scored later must match, or predict_proba sees a different
    distribution than training.
    """
    df = df.copy()
    df["is_dependency_of_sink"] = df["is_dependency_of_sink"].astype(int)
    df = df.fillna(MISSING_VALUE_SENTINEL)
    return df


def load_features(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    df = clean_feature_values(df)
    return df


def split_by_trace(df, random_seed=RANDOM_SEED, test_fraction=1 / 3):
    trace_ids = df["trace_id"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(trace_ids)

    n_test = max(1, round(len(shuffled) * test_fraction))
    n_test = min(n_test, len(shuffled) - 1)  # always leave >= 1 trace for train

    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])

    train_df = df[df["trace_id"].isin(train_ids)].drop(columns=["trace_id"])
    test_df = df[df["trace_id"].isin(test_ids)].drop(columns=["trace_id"])
    return train_df, test_df, sorted(train_ids), sorted(test_ids)


def safe_auc(y_true, y_score, metric_fn, metric_name):
    if len(set(y_true)) < 2:
        return f"undefined ({metric_name}: test set has only one class)"
    return f"{metric_fn(y_true, y_score):.4f}"


def cross_validate(df, feature_names, extra_df=None, n_splits=N_CV_SPLITS, random_seed=RANDOM_SEED):
    """
    GroupKFold cross-validation grouped by trace_id: rows within one trace are correlated, so a trace's rows
    must never span both the train and validation side of a fold, or that's
    leakage into the validation score.

    fits a fresh baseline + primary model per fold, using each row's
    sample weight (df[WEIGHT_COLUMN], all 1.0 unless --extra-input was
    used) so the training-only up-weighting matches what main()'s single
    split does. EVALUATION (AUC/PR-AUC, and the calibration check that
    reuses these out-of-fold probabilities) stays unweighted, so reported
    numbers reflect real observed rates on real data, not a rebalanced view.

    if extra_df is given (e.g. sweep training rows), its rows are added to
    the TRAINING side of every fold (never the validation side, and never
    counted in oof_probs) -- same "train-only augmentation, never part of
    the evaluated distribution" rule main()'s single split follows, kept
    consistent here so CV isn't secretly evaluating on a mix of
    distributions.

    also collects each row's out-of-fold predicted probability (the
    prediction made on it by the one fold where it was in the validation
    split, i.e. never trained on) for later use in the calibration check --
    rows in a skipped fold (validation split had only one class) are left
    as NaN, since no valid prediction exists for them.
    """
    X = df[feature_names]
    y = df[TARGET_COLUMN]
    w = df[WEIGHT_COLUMN]
    groups = df["trace_id"]

    n_traces = groups.nunique()
    n_splits = min(n_splits, n_traces)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    results = {
        "logistic regression": {"auc_roc": [], "pr_auc": []},
        PRIMARY_MODEL_NAME: {"auc_roc": [], "pr_auc": []},
    }
    oof_probs = {
        "logistic regression": np.full(len(df), np.nan),
        PRIMARY_MODEL_NAME: np.full(len(df), np.nan),
    }

    for fold_index, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        w_train_fold = w.iloc[train_idx]

        if extra_df is not None:
            X_train_fold = pd.concat([X_train_fold, extra_df[feature_names]], ignore_index=True)
            y_train_fold = pd.concat([y_train_fold, extra_df[TARGET_COLUMN]], ignore_index=True)
            w_train_fold = pd.concat([w_train_fold, extra_df[WEIGHT_COLUMN]], ignore_index=True)

        if len(set(y_val_fold)) < 2:
            print(f"  fold {fold_index}/{n_splits}: skipped (validation fold has only one class)")
            continue

        baseline_fold = LogisticRegression(max_iter=5000, class_weight="balanced")
        baseline_fold.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
        baseline_scores = baseline_fold.predict_proba(X_val_fold)[:, 1]
        results["logistic regression"]["auc_roc"].append(roc_auc_score(y_val_fold, baseline_scores))
        results["logistic regression"]["pr_auc"].append(average_precision_score(y_val_fold, baseline_scores))
        oof_probs["logistic regression"][val_idx] = baseline_scores

        primary_fold = XGBClassifier(n_estimators=100, max_depth=3, random_state=random_seed, eval_metric="logloss")
        primary_fold.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
        primary_scores = primary_fold.predict_proba(X_val_fold)[:, 1]
        results[PRIMARY_MODEL_NAME]["auc_roc"].append(roc_auc_score(y_val_fold, primary_scores))
        results[PRIMARY_MODEL_NAME]["pr_auc"].append(average_precision_score(y_val_fold, primary_scores))
        oof_probs[PRIMARY_MODEL_NAME][val_idx] = primary_scores

        print(
            f"  fold {fold_index}/{n_splits}: "
            f"logreg AUC-ROC={results['logistic regression']['auc_roc'][-1]:.4f} PR-AUC={results['logistic regression']['pr_auc'][-1]:.4f}  |  "
            f"{PRIMARY_MODEL_NAME} AUC-ROC={results[PRIMARY_MODEL_NAME]['auc_roc'][-1]:.4f} PR-AUC={results[PRIMARY_MODEL_NAME]['pr_auc'][-1]:.4f}"
        )

    return results, oof_probs


def _mean_std(values: list) -> str:
    if not values:
        return "undefined (no valid folds)"
    arr = np.array(values)
    return f"{arr.mean():.4f} +/- {arr.std():.4f} (n={len(arr)} folds)"


N_CALIBRATION_BUCKETS = 10


def calibration_table(y_true, y_prob, n_buckets=N_CALIBRATION_BUCKETS):
    """
    buckets y_prob into n_buckets equal-width bins (0.0-0.1, 0.1-0.2, ...,
    0.9-1.0) and reports, per bucket: how many predictions landed there,
    the model's own mean predicted probability within the bucket, and the
    actual observed will_be_reused rate within the bucket. empty buckets
    are included with count=0 rather than skipped, so gaps in coverage
    (e.g. no predictions above 60%) are visible, not hidden.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bucket_idx = np.minimum((y_prob * n_buckets).astype(int), n_buckets - 1)

    rows = []
    for b in range(n_buckets):
        mask = bucket_idx == b
        count = int(mask.sum())
        rows.append(
            {
                "bucket": f"{b / n_buckets:.1f}-{(b + 1) / n_buckets:.1f}",
                "count": count,
                "mean_predicted": float(y_prob[mask].mean()) if count else None,
                "observed_reuse_rate": float(y_true[mask].mean()) if count else None,
            }
        )
    return rows


def print_calibration_table(title: str, rows: list, brier_score: float) -> None:
    print(f"  {title} (Brier score: {brier_score:.4f}, lower is better -- 0 is perfect)")
    print(f"    {'bucket':<12}{'count':>8}{'mean predicted':>18}{'observed rate':>16}")
    for r in rows:
        mean_pred = f"{r['mean_predicted']:.3f}" if r["mean_predicted"] is not None else "n/a"
        obs_rate = f"{r['observed_reuse_rate']:.3f}" if r["observed_reuse_rate"] is not None else "n/a"
        print(f"    {r['bucket']:<12}{r['count']:>8}{mean_pred:>18}{obs_rate:>16}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features.csv", help="path to features CSV")
    parser.add_argument(
        "--extra-input",
        default=None,
        help="optional second features CSV folded into TRAINING ONLY (never the held-out test split or any CV validation fold), up-weighted so it isn't drowned out by raw row count -- e.g. features_sweep.csv",
    )
    parser.add_argument(
        "--extra-weight-fraction",
        type=float,
        default=DEFAULT_EXTRA_WEIGHT_FRACTION,
        help="target fraction of total training weight contributed by --extra-input rows",
    )
    parser.add_argument("--model-out", default="model.joblib", help="path to write the trained model")
    args = parser.parse_args()

    df = load_features(args.input)
    df[WEIGHT_COLUMN] = 1.0
    feature_names = [c for c in df.columns if c not in ("trace_id", TARGET_COLUMN, WEIGHT_COLUMN)]

    n_traces = df["trace_id"].nunique()
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    positive_rate = df[TARGET_COLUMN].mean()

    train_df, test_df, train_ids, test_ids = split_by_trace(df)

    extra_df = None
    extra_weight = None
    extra_trace_ids = []
    if args.extra_input:
        extra_df = load_features(args.extra_input)
        overlap = set(df["trace_id"]) & set(extra_df["trace_id"])
        if overlap:
            raise ValueError(f"--extra-input shares trace_id(s) with --input, refusing to combine: {overlap}")
        frac = args.extra_weight_fraction
        extra_weight = (frac * len(train_df)) / ((1 - frac) * len(extra_df))
        extra_df[WEIGHT_COLUMN] = extra_weight
        extra_trace_ids = sorted(extra_df["trace_id"].unique())

    X_train, y_train, w_train = train_df[feature_names], train_df[TARGET_COLUMN], train_df[WEIGHT_COLUMN]
    if extra_df is not None:
        X_train = pd.concat([X_train, extra_df[feature_names]], ignore_index=True)
        y_train = pd.concat([y_train, extra_df[TARGET_COLUMN]], ignore_index=True)
        w_train = pd.concat([w_train, extra_df[WEIGHT_COLUMN]], ignore_index=True)
    # test_df is built from args.input ONLY (via split_by_trace above, before
    # any extra_df concatenation) -- extra rows never enter evaluation, so
    # this stays the exact same held-out set previously-reported general
    # benchmark numbers were computed against, whether or not --extra-input
    # is used
    X_test, y_test = test_df[feature_names], test_df[TARGET_COLUMN]

    baseline = LogisticRegression(max_iter=5000, class_weight="balanced")
    baseline.fit(X_train, y_train, sample_weight=w_train)
    baseline_scores = baseline.predict_proba(X_test)[:, 1]
    baseline_auc_roc = safe_auc(y_test, baseline_scores, roc_auc_score, "AUC-ROC")
    baseline_pr_auc = safe_auc(y_test, baseline_scores, average_precision_score, "PR-AUC")

    primary = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
    )
    primary.fit(X_train, y_train, sample_weight=w_train)
    primary_scores = primary.predict_proba(X_test)[:, 1]
    primary_auc_roc = safe_auc(y_test, primary_scores, roc_auc_score, "AUC-ROC")
    primary_pr_auc = safe_auc(y_test, primary_scores, average_precision_score, "PR-AUC")

    joblib.dump({"model": primary, "feature_names": feature_names}, args.model_out)

    print(f"\nrunning {N_CV_SPLITS}-fold GroupKFold cross-validation (group=trace_id)...")
    cv_results, oof_probs = cross_validate(df, feature_names, extra_df=extra_df)

    print("=" * 70)
    print("PIPELINE CORRECTNESS CHECK -- NOT A REAL RESULT")
    print(f"input file: {args.input}")
    if extra_df is not None:
        print(
            f"extra training input: {args.extra_input} ({len(extra_trace_ids)} traces, {len(extra_df)} rows, "
            f"weight={extra_weight:.3f}/row, targeting {args.extra_weight_fraction:.0%} of training weight)"
        )
        print(f"extra trace_ids: {extra_trace_ids}")
    print("=" * 70)
    print(f"dataset: {len(df)} rows across {n_traces} traces")
    print(f"class balance ({TARGET_COLUMN}): {dict(class_counts)} (positive rate: {positive_rate:.3f})")
    print(f"train traces ({len(train_ids)}): {train_ids}")
    print(f"test traces  ({len(test_ids)}): {test_ids}")
    print(f"baseline (logistic regression) coefficients:")
    for name, coef in zip(feature_names, baseline.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    print(f"--- single train/test split ---")
    print(f"baseline AUC-ROC: {baseline_auc_roc}  |  baseline PR-AUC: {baseline_pr_auc}")
    print(f"primary  ({PRIMARY_MODEL_NAME}) AUC-ROC: {primary_auc_roc}  |  primary PR-AUC: {primary_pr_auc}")
    print(f"--- {N_CV_SPLITS}-fold GroupKFold cross-validation (group=trace_id) ---")
    print(f"baseline AUC-ROC: {_mean_std(cv_results['logistic regression']['auc_roc'])}")
    print(f"baseline PR-AUC:  {_mean_std(cv_results['logistic regression']['pr_auc'])}")
    print(f"primary  ({PRIMARY_MODEL_NAME}) AUC-ROC: {_mean_std(cv_results[PRIMARY_MODEL_NAME]['auc_roc'])}")
    print(f"primary  ({PRIMARY_MODEL_NAME}) PR-AUC:  {_mean_std(cv_results[PRIMARY_MODEL_NAME]['pr_auc'])}")
    print(f"--- calibration check (out-of-fold predictions from cross-validation above) ---")
    y_all = df[TARGET_COLUMN].to_numpy()
    for model_name in ("logistic regression", PRIMARY_MODEL_NAME):
        probs = oof_probs[model_name]
        valid = ~np.isnan(probs)
        n_missing = int((~valid).sum())
        missing_note = f" [{n_missing} rows excluded: no out-of-fold prediction, from a skipped fold]" if n_missing else ""
        if valid.sum() == 0:
            print(f"  {model_name}: no valid out-of-fold predictions to calibrate against{missing_note}")
            continue
        table = calibration_table(y_all[valid], probs[valid])
        brier = brier_score_loss(y_all[valid], probs[valid])
        print_calibration_table(f"{model_name}{missing_note}", table, brier)
    print(f"model saved to: {args.model_out}")
    print("=" * 70)
    print(
        "DISCLAIMER: this run is a pipeline correctness check on a too-small "
        "dataset (and/or synthetic data), not a real evaluation. These "
        "numbers are not statistically meaningful. Re-run once a larger "
        "batch of real trace data exists."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
