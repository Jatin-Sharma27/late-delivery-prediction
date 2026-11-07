"""Run the model chosen by the pre-test backtest once on the test period.

The choice (xgb_core_only with fixed settings) came from src/diagnose.py.
This is not a blind test: the test period had already been seen once, in the
first attempt recorded in reports/first_attempt_metrics.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from diagnose import xgb
from features import CATEGORICAL, NUMERIC, load_dataset
from train import SEED, TEST_START, VAL_START, bootstrap_diff, evaluate, make_logreg, time_split

REPORTS = Path(__file__).resolve().parents[1] / "reports"
CORE = ["slack_days", "est_days", "route_mean_days", "distance_km", "same_state"]


def main():
    df = load_dataset()
    train, val, test = time_split(df)
    final_train = pd.concat([train, val])
    y = test.is_late.to_numpy()

    model, cols = xgb(CORE, [])
    model.fit(final_train[cols], final_train.is_late)
    p_xgb = model.predict_proba(test[cols])[:, 1]

    logreg = make_logreg().fit(final_train[NUMERIC + CATEGORICAL], final_train.is_late)
    p_log = logreg.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
    rule = -test["slack_days"].fillna(test["slack_days"].median()).to_numpy()

    months = test.purchase_ts.str[:7].to_numpy()
    results = {
        "note": "Second look at the test period, not a blind test.",
        "orders": len(y), "late_orders": int(y.sum()), "late_rate": float(y.mean()),
        "train_orders": len(final_train),
        "constant_baseline_pr_auc": float(y.mean()),
        "slack_rule": evaluate(y, rule, probabilities=False),
        "logistic_regression": evaluate(y, p_log),
        "xgboost_core": evaluate(y, p_xgb),
        "constant_baseline_brier": float(brier_score_loss(y, np.full(len(y), final_train.is_late.mean()))),
        "ci95_pr_auc_diff": {"xgb_minus_rule": bootstrap_diff(y, p_xgb, rule),
                             "xgb_minus_logreg": bootstrap_diff(y, p_xgb, p_log)},
        "by_month": {m: {"orders": int((months == m).sum()),
                         "late_rate": float(y[months == m].mean()),
                         "xgb_pr_auc": float(average_precision_score(y[months == m], p_xgb[months == m])),
                         "rule_pr_auc": float(average_precision_score(y[months == m], rule[months == m]))}
                     for m in sorted(set(months))},
    }
    (REPORTS / "final_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, score in [("XGBoost (core features)", p_xgb), ("Logistic regression", p_log), ("Slack rule", rule)]:
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=f"{label} (PR-AUC {average_precision_score(y, score):.3f})")
    ax.axhline(y.mean(), ls="--", c="grey", label=f"Late rate {y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test period: Jun-Aug 2018")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "pr_curve.png", dpi=150)

    # Promised window shrinking: the drift behind the first attempt's failure.
    monthly = df[df.purchase_ts >= "2017-09"].assign(month=lambda d: d.purchase_ts.str[:7]) \
        .groupby("month")["est_days"].median()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    monthly.plot(ax=ax, marker="o")
    ax.set(xlabel="Purchase month", ylabel="Median promised days", title="The delivery promise shifted before the test period")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS / "promise_drift.png", dpi=150)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
