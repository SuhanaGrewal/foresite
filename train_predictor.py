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

TODO(more data):
- once there are enough predicted-probability buckets with enough rows
  each, add a calibration reliability check (e.g. a reliability diagram
  / Brier score by bucket). Skipped for now because that needs more data
  per bucket than a plain train/test split or k-fold check does.
"""

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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

NON_FEATURE_COLUMNS = ID_COLUMNS + ["event_type"]

MISSING_VALUE_SENTINEL = -1


def load_features(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    df["is_dependency_of_sink"] = df["is_dependency_of_sink"].astype(int)
    df = df.fillna(MISSING_VALUE_SENTINEL)
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


def cross_validate(df, feature_names, n_splits=N_CV_SPLITS, random_seed=RANDOM_SEED):
    """
    GroupKFold cross-validation grouped by trace_id: rows within one trace are correlated, so a trace's rows
    must never span both the train and validation side of a fold, or that's
    leakage into the validation score.

    fits a fresh baseline + primary model per fold.
    """
    X = df[feature_names]
    y = df[TARGET_COLUMN]
    groups = df["trace_id"]

    n_traces = groups.nunique()
    n_splits = min(n_splits, n_traces)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    results = {
        "logistic regression": {"auc_roc": [], "pr_auc": []},
        PRIMARY_MODEL_NAME: {"auc_roc": [], "pr_auc": []},
    }

    for fold_index, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        if len(set(y_val_fold)) < 2:
            print(f"  fold {fold_index}/{n_splits}: skipped (validation fold has only one class)")
            continue

        baseline_fold = LogisticRegression(max_iter=5000, class_weight="balanced")
        baseline_fold.fit(X_train_fold, y_train_fold)
        baseline_scores = baseline_fold.predict_proba(X_val_fold)[:, 1]
        results["logistic regression"]["auc_roc"].append(roc_auc_score(y_val_fold, baseline_scores))
        results["logistic regression"]["pr_auc"].append(average_precision_score(y_val_fold, baseline_scores))

        primary_fold = XGBClassifier(n_estimators=100, max_depth=3, random_state=random_seed, eval_metric="logloss")
        primary_fold.fit(X_train_fold, y_train_fold)
        primary_scores = primary_fold.predict_proba(X_val_fold)[:, 1]
        results[PRIMARY_MODEL_NAME]["auc_roc"].append(roc_auc_score(y_val_fold, primary_scores))
        results[PRIMARY_MODEL_NAME]["pr_auc"].append(average_precision_score(y_val_fold, primary_scores))

        print(
            f"  fold {fold_index}/{n_splits}: "
            f"logreg AUC-ROC={results['logistic regression']['auc_roc'][-1]:.4f} PR-AUC={results['logistic regression']['pr_auc'][-1]:.4f}  |  "
            f"{PRIMARY_MODEL_NAME} AUC-ROC={results[PRIMARY_MODEL_NAME]['auc_roc'][-1]:.4f} PR-AUC={results[PRIMARY_MODEL_NAME]['pr_auc'][-1]:.4f}"
        )

    return results


def _mean_std(values: list) -> str:
    if not values:
        return "undefined (no valid folds)"
    arr = np.array(values)
    return f"{arr.mean():.4f} +/- {arr.std():.4f} (n={len(arr)} folds)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features.csv", help="path to features CSV")
    parser.add_argument("--model-out", default="model.joblib", help="path to write the trained model")
    args = parser.parse_args()

    df = load_features(args.input)
    feature_names = [c for c in df.columns if c not in ("trace_id", TARGET_COLUMN)]

    n_traces = df["trace_id"].nunique()
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    positive_rate = df[TARGET_COLUMN].mean()

    train_df, test_df, train_ids, test_ids = split_by_trace(df)

    X_train, y_train = train_df[feature_names], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_names], test_df[TARGET_COLUMN]

    baseline = LogisticRegression(max_iter=5000, class_weight="balanced")
    baseline.fit(X_train, y_train)
    baseline_scores = baseline.predict_proba(X_test)[:, 1]
    baseline_auc_roc = safe_auc(y_test, baseline_scores, roc_auc_score, "AUC-ROC")
    baseline_pr_auc = safe_auc(y_test, baseline_scores, average_precision_score, "PR-AUC")

    primary = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
    )
    primary.fit(X_train, y_train)
    primary_scores = primary.predict_proba(X_test)[:, 1]
    primary_auc_roc = safe_auc(y_test, primary_scores, roc_auc_score, "AUC-ROC")
    primary_pr_auc = safe_auc(y_test, primary_scores, average_precision_score, "PR-AUC")

    joblib.dump({"model": primary, "feature_names": feature_names}, args.model_out)

    print(f"\nrunning {N_CV_SPLITS}-fold GroupKFold cross-validation (group=trace_id)...")
    cv_results = cross_validate(df, feature_names)

    print("=" * 70)
    print("PIPELINE CORRECTNESS CHECK -- NOT A REAL RESULT")
    print(f"input file: {args.input}")
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
