"""Find out why the first XGBoost failed, using pre-test outcomes only.

Rolling-origin backtest: each fold trains on the past and scores the next
three months. Nothing here reads outcomes from June 2018 onwards.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import CATEGORICAL, NUMERIC, load_dataset
from train import SEED, TEST_START, make_logreg, preprocessor

warnings.filterwarnings("ignore")
FOLDS = [("2017-09-01", "2017-12-01"), ("2017-12-01", "2018-03-01"), ("2018-03-01", "2018-06-01")]
REGIME = ["network_late_rate_30d"]
RATES = ["seller_late_rate", "route_late_rate"]


def fold(df, start, end):
    tr = df[(df.purchase_ts >= "2017-01-01") & (df.purchase_ts < start) & (df.delivered_ts < start)]
    va = df[(df.purchase_ts >= start) & (df.purchase_ts < end) & (df.delivered_ts < TEST_START)]
    return tr, va


def xgb(num, cat, **kw):
    import train as t
    prep = t.ColumnTransformer([
        ("num", t.make_pipeline(t.SimpleImputer(strategy="median")), num),
        ("cat", t.OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat),
    ]) if cat else t.ColumnTransformer([("num", t.make_pipeline(t.SimpleImputer(strategy="median")), num)])
    params = dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1)
    params.update(kw)
    return Pipeline([("prep", prep), ("model", XGBClassifier(**params))]), num + cat


def main():
    df = load_dataset()
    df["month"] = df.purchase_ts.str[:7]

    print("Inputs by month (test months have no outcomes shown):")
    cols = ["est_days", "route_mean_days", "slack_days", "network_late_rate_30d", "route_late_rate"]
    print(df[df.month >= "2017-09"].groupby("month")[cols].median().round(3).to_string())

    print("\nOutcomes by month, pre-test only:")
    pre = df[df.purchase_ts < TEST_START]
    print(pre[pre.month >= "2017-09"].groupby("month").agg(late=("is_late", "mean"),
          actual_days=("actual_days", "median")).round(3).to_string())

    no_regime = [c for c in NUMERIC if c not in REGIME]
    no_rates = [c for c in no_regime if c not in RATES]
    core = ["slack_days", "est_days", "route_mean_days", "distance_km", "same_state"]
    configs = {
        "xgb_all_features": xgb(NUMERIC, CATEGORICAL),
        "xgb_no_network_rate": xgb(no_regime, CATEGORICAL),
        "xgb_no_rate_features": xgb(no_rates, CATEGORICAL),
        "xgb_core_only": xgb(core, []),
    }
    rows = []
    for start, end in FOLDS:
        tr, va = fold(df, start, end)
        y = va.is_late.to_numpy()
        for name, (model, cols) in configs.items():
            model.fit(tr[cols], tr.is_late)
            p = model.predict_proba(va[cols])[:, 1]
            rows.append((start, name, average_precision_score(y, p), roc_auc_score(y, p)))
        lr = make_logreg().fit(tr[NUMERIC + CATEGORICAL], tr.is_late)
        p = lr.predict_proba(va[NUMERIC + CATEGORICAL])[:, 1]
        rows.append((start, "logistic_regression", average_precision_score(y, p), roc_auc_score(y, p)))
        rule = -va.slack_days.fillna(va.slack_days.median()).to_numpy()
        rows.append((start, "slack_rule", average_precision_score(y, rule), roc_auc_score(y, rule)))
        rows.append((start, "late_rate", y.mean(), 0.5))
    res = pd.DataFrame(rows, columns=["fold", "model", "pr_auc", "roc_auc"])
    print("\nPR-AUC by fold:")
    print(res.pivot(index="model", columns="fold", values="pr_auc").round(3).to_string())
    print("\nROC-AUC by fold:")
    print(res.pivot(index="model", columns="fold", values="roc_auc").round(3).to_string())
    print("\nMean across folds:")
    print(res.groupby("model")[["pr_auc", "roc_auc"]].mean().round(3).sort_values("pr_auc", ascending=False).to_string())


if __name__ == "__main__":
    main()
