#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import shap
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from .feature_engineering import CANDIDATE_FEATURES
from .utils import processed_shap_path, MODELS_DIR, ensure_dir
import os


def train_xgb_classifier(df: pd.DataFrame) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train an XGBoost binary classifier on CANDIDATE_FEATURES vs is_anomaly_label.
    Returns model + train/val split.
    """
    df = df.copy()
    max_rows_for_xgb = min(20000, len(df))
    xgb_df = df.sample(n=max_rows_for_xgb, random_state=42).copy()

    X = xgb_df[CANDIDATE_FEATURES]
    y = xgb_df["is_anomaly_label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgbmodel = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        verbosity=0,
        random_state=42
    )

    xgbmodel.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_val = xgbmodel.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds_val)
    print(f"✅ XGBoost validation AUC: {auc:.4f}")

    preds_bin = (preds_val >= 0.5).astype(int)
    print(classification_report(y_val, preds_bin, zero_division=0))

    return xgbmodel, X_train, y_train, X_val, y_val


def compute_shap_for_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    full_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute SHAP values with a model.predict_proba wrapper and attach local
    SHAP drivers to full_df (per-row).
    """
    # Background
    background = X_train.sample(
        n=min(2000, len(X_train)),
        random_state=42
    ).astype("float64")

    def model_predict_proba(X_input):
        return model.predict_proba(X_input)[:, 1]

    explainer = shap.Explainer(model_predict_proba, background)

    # Global SHAP on a validation sample
    sample_X = X_val.sample(n=min(4000, len(X_val)), random_state=42).astype("float64")
    shap_result = explainer(sample_X)
    shap_values = shap_result.values  # (n_samples, n_features)
    print("SHAP matrix shape:", shap_values.shape)

    shap_global = (
        pd.DataFrame({
            "feature": sample_X.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("Top 10 SHAP features:\n", shap_global.head(10))

    # Local SHAP for all rows we want to enrich
    df_for_shap = full_df[CANDIDATE_FEATURES].astype("float64")
    shap_full = explainer(df_for_shap).values  # (N, n_features)

    def top_local_features(shap_row, n=4):
        abs_vals = np.abs(shap_row)
        idx = np.argsort(-abs_vals)[:n]
        return [CANDIDATE_FEATURES[i] for i in idx]

    full_df = full_df.copy()
    full_df["top_shap_features"] = [
        top_local_features(shap_full[i], n=4)
        for i in range(len(shap_full))
    ]

    # Optional: a human-readable SHAP summary sentence
    def row_shap_summary(shap_row):
        abs_vals = np.abs(shap_row)
        idx = np.argsort(-abs_vals)[:4]
        parts = []
        for i in idx:
            feat = CANDIDATE_FEATURES[i]
            val = shap_row[i]
            sign = "+" if val >= 0 else "-"
            parts.append(f"{feat} {sign}{abs(val):.3f}")
        return "Top factors: " + ", ".join(parts) + "."

    full_df["shap_summary"] = [
        row_shap_summary(shap_full[i])
        for i in range(len(shap_full))
    ]

    return shap_global, full_df


def run_xgb_and_shap():
    """
    Orchestrator: load processed data, train XGB, compute SHAP,
    and save artifacts.
    """
    ensure_dir(MODELS_DIR)
    df = pd.read_csv(processed_shap_path())

    model, X_train, y_train, X_val, y_val = train_xgb_classifier(df)
    shap_global, df_with_shap = compute_shap_for_model(model, X_train, X_val, df)

    # Save model & SHAP outputs
    model_path = os.path.join(MODELS_DIR, "xgb_anomaly_model.json")
    model.save_model(model_path)
    print(f"Saved XGBoost model to {model_path}")

    shap_global_path = os.path.join(MODELS_DIR, "shap_global_top20.csv")
    shap_global.head(20).to_csv(shap_global_path, index=False)
    print(f"Saved SHAP global importance to {shap_global_path}")

    # Save enriched dataframe for SLM
    from .utils import processed_shap_path as enriched_path
    df_with_shap.to_csv(enriched_path(), index=False)
    print(f"Saved SHAP-enriched dataset to {enriched_path()}, shape {df_with_shap.shape}")

    return df_with_shap, shap_global


if __name__ == "__main__":
    run_xgb_and_shap()

