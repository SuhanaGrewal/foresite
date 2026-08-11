"""
Train Predictor

Trains a cache-reuse predictor from a features CSV produced by
feature_extraction.py (schema: trace_id, row_index, item_id, agent_id,
parent_id, event_type, timestamp, ... will_be_reused, event_type_*).

*** PIPELINE CORRECTNESS CHECK, NOT A REAL RESULT ***
As of this writing there are only a handful of real traces (well under 50
rows total), and the default run uses entirely synthetic data from
generate_fake_features.py. AUC-ROC / PR-AUC numbers printed by this script
are not statistically meaningful at this sample size and must not be quoted
as a real evaluation of the model. Re-run against a larger, real
features.csv before drawing any conclusions.

Two models are trained:
  - logistic regression: baseline, coefficients printed for inspection
  - XGBoost: primary model

The train/test split is done at the TRACE level (never row level), since
rows within one trace are correlated -- a row-level split would leak
information from a trace's other rows into the test set.

TODO(more data): once there are enough traces to do a meaningful
GroupKFold split (group = trace_id), add k-fold cross-validation here.
Skipped for now because with a handful of traces, folds would be tiny and
their variance would be uninterpretable.

TODO(more data): once there are enough predicted-probability buckets with
enough rows each, add a calibration reliability check (e.g. a reliability
diagram / Brier score by bucket). Skipped for now for the same reason.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from xgboost import XGBClassifier

    PRIMARY_MODEL_NAME = "xgboost"
except ImportError:
    from lightgbm import LGBMClassifier as XGBClassifier

    PRIMARY_MODEL_NAME = "lightgbm"

RANDOM_SEED = 42
TARGET_COLUMN = "will_be_reused"
ID_COLUMNS = ["item_id", "timestamp", "agent_id", "parent_id"]
# event_type is dropped too: it's already one-hot encoded as
# event_type_model_call / event_type_tool_call / event_type_final_answer,
# and as a raw string it isn't a usable model input.
NON_FEATURE_COLUMNS = ID_COLUMNS + ["event_type"]
# -1 is the same "not applicable / no prior value" sentinel steps_since_last_seen
# and seconds_since_last_seen already use, so missing measured_latency_seconds
# (agent's first record) and dag_completion_fraction (legacy traces with no DAG)
# reuse that convention instead of introducing a second missing-value scheme.
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
    print(f"baseline AUC-ROC: {baseline_auc_roc}  |  baseline PR-AUC: {baseline_pr_auc}")
    print(f"primary  ({PRIMARY_MODEL_NAME}) AUC-ROC: {primary_auc_roc}  |  primary PR-AUC: {primary_pr_auc}")
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
