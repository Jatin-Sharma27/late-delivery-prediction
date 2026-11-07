"""First attempt: all features, tuned on validation. It failed on the test period.

Kept so the failure can be reproduced. See src/diagnose.py and src/final_eval.py.
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CATEGORICAL, NUMERIC, load_dataset

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 42
TRAIN_START, VAL_START, TEST_START = "2017-01-01", "2018-03-01", "2018-06-01"
FEATURES = NUMERIC + CATEGORICAL


def time_split(df):
    """Split by purchase date. An order's outcome must be known before the next
    period starts, so orders still in transit at that point are dropped."""
    train = df[(df.purchase_ts >= TRAIN_START) & (df.purchase_ts < VAL_START)
               & (df.delivered_ts < VAL_START)]
    val = df[(df.purchase_ts >= VAL_START) & (df.purchase_ts < TEST_START)
             & (df.delivered_ts < TEST_START)]
    test = df[df.purchase_ts >= TEST_START]
    return train, val, test


def preprocessor(scale):
    numeric = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([
        ("num", make_pipeline(*numeric), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAL),
    ])


def top_k_stats(y, score, fraction=0.10):
    k = max(1, int(len(y) * fraction))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean()), float(y[top].sum() / y.sum())


def evaluate(y, score, probabilities=True):
    precision, recall = top_k_stats(y, score)
    out = {"pr_auc": float(average_precision_score(y, score)),
           "roc_auc": float(roc_auc_score(y, score)),
           "precision_top10pct": precision,
           "recall_top10pct": recall}
    if probabilities:
        out["brier"] = float(brier_score_loss(y, score))
    return out


def bootstrap_diff(y, a, b, n=1000):
    """95% interval for the PR-AUC of a minus b on the test set."""
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def tune_xgb(train, val):
    prep = preprocessor(scale=False).fit(train[FEATURES])
    x_train, x_val = prep.transform(train[FEATURES]), prep.transform(val[FEATURES])
    grid = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1],
            "min_child_weight": [1, 10], "subsample": [0.8]}
    best = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        model = XGBClassifier(n_estimators=1500, colsample_bytree=0.8, tree_method="hist",
                              eval_metric="aucpr", early_stopping_rounds=50,
                              random_state=SEED, n_jobs=-1, **params)
        model.fit(x_train, train.is_late, eval_set=[(x_val, val.is_late)], verbose=False)
        score = average_precision_score(val.is_late, model.predict_proba(x_val)[:, 1])
        if best is None or score > best["val_pr_auc"]:
            best = {"params": params, "n_estimators": int(model.best_iteration) + 1,
                    "val_pr_auc": float(score)}
    return best


def make_xgb(params, n_estimators):
    return Pipeline([("prep", preprocessor(scale=False)),
                     ("model", XGBClassifier(n_estimators=n_estimators, colsample_bytree=0.8,
                                             tree_method="hist", random_state=SEED, n_jobs=-1,
                                             **params))])


def make_logreg():
    return Pipeline([("prep", preprocessor(scale=True)),
                     ("model", LogisticRegression(max_iter=2000, C=1.0))])


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_dataset()
    train, val, test = time_split(df)
    y_val, y_test = val.is_late.to_numpy(), test.is_late.to_numpy()
    print(f"train {len(train):,} val {len(val):,} test {len(test):,}")

    best = tune_xgb(train, val)
    print("best on validation:", best)

    # Compare models on validation first, using training data only.
    val_scores = {}
    logreg = make_logreg().fit(train[FEATURES], train.is_late)
    val_scores["logistic_regression"] = evaluate(y_val, logreg.predict_proba(val[FEATURES])[:, 1])
    xgb = make_xgb(best["params"], best["n_estimators"]).fit(train[FEATURES], train.is_late)
    val_scores["xgboost"] = evaluate(y_val, xgb.predict_proba(val[FEATURES])[:, 1])

    # Refit on everything whose outcome was known before the test period.
    final_train = pd.concat([train, val])
    logreg = make_logreg().fit(final_train[FEATURES], final_train.is_late)
    xgb = make_xgb(best["params"], best["n_estimators"]).fit(final_train[FEATURES], final_train.is_late)

    p_xgb = xgb.predict_proba(test[FEATURES])[:, 1]
    p_log = logreg.predict_proba(test[FEATURES])[:, 1]
    # Simple rule: orders with the least margin over the usual route time go first.
    rule = -test["slack_days"].fillna(test["slack_days"].median()).to_numpy()
    p_prior = np.full(len(y_test), final_train.is_late.mean())

    results = {
        "split": {"train": len(train), "val": len(val), "test": len(test),
                  "final_train": len(final_train),
                  "late_rate_train": float(train.is_late.mean()),
                  "late_rate_val": float(val.is_late.mean()),
                  "late_rate_test": float(y_test.mean()),
                  "test_late_orders": int(y_test.sum())},
        "best_xgb": best,
        "validation": val_scores,
        "test": {
            "constant_baseline": {"pr_auc": float(y_test.mean()), "roc_auc": 0.5,
                                  "precision_top10pct": float(y_test.mean()),
                                  "recall_top10pct": 0.10,
                                  "brier": float(brier_score_loss(y_test, p_prior))},
            "slack_rule": evaluate(y_test, rule, probabilities=False),
            "logistic_regression": evaluate(y_test, p_log),
            "xgboost": evaluate(y_test, p_xgb),
        },
        "ci95_pr_auc_diff": {"xgboost_minus_logreg": bootstrap_diff(y_test, p_xgb, p_log),
                             "xgboost_minus_rule": bootstrap_diff(y_test, p_xgb, rule)},
    }

    months = test.purchase_ts.str[:7].to_numpy()
    results["test_by_month"] = {
        m: {"orders": int((months == m).sum()), "late_rate": float(y_test[months == m].mean()),
            "xgb_pr_auc": float(average_precision_score(y_test[months == m], p_xgb[months == m]))}
        for m in sorted(set(months))
    }

    # Importance is measured on validation with a train-only model, so the test
    # set plays no part in how the model is described.
    train_only = make_xgb(best["params"], best["n_estimators"]).fit(train[FEATURES], train.is_late)
    imp = permutation_importance(train_only, val[FEATURES], y_val, scoring="average_precision",
                                 n_repeats=5, random_state=SEED, n_jobs=1)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
    results["importance_val"] = {k: float(v) for k, v in importance.head(10).items()}

    (REPORTS / "first_attempt_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, score in [("XGBoost", p_xgb), ("Logistic regression", p_log), ("Slack rule", rule)]:
        precision, recall, _ = precision_recall_curve(y_test, score)
        ax.plot(recall, precision, label=f"{label} (PR-AUC {average_precision_score(y_test, score):.3f})")
    ax.axhline(y_test.mean(), ls="--", c="grey", label=f"Late rate {y_test.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test set: Jun-Aug 2018")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "first_attempt_pr_curve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    importance.head(10).iloc[::-1].plot.barh(ax=ax)
    ax.set(xlabel="Drop in validation PR-AUC when shuffled", title="Most useful features")
    fig.tight_layout()
    fig.savefig(REPORTS / "first_attempt_feature_importance.png", dpi=150)

    print(json.dumps({k: results[k] for k in
                      ["split", "best_xgb", "validation", "test", "ci95_pr_auc_diff",
                       "test_by_month", "importance_val"]}, indent=1))


if __name__ == "__main__":
    main()
