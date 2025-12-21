#!/usr/bin/env python
# coding: utf-8

import os
from typing import Tuple

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from .feature_engineering import CANDIDATE_FEATURES
from .utils import processed_fe_path, processed_shap_path, MODELS_DIR, ensure_dir


def train_xgb_classifier(
    df: pd.DataFrame,
) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train an XGBoost binary classifier on CANDIDATE_FEATURES vs is_anomaly_label.
    Returns model + train/val split.
    """
    df = df.copy()

    # For dev speed, sample up to 20k rows
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
        random_state=42,
    )

    xgbmodel.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
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
    full_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute SHAP values using a model.predict_proba wrapper and attach
    per-row SHAP drivers + SHAP summary strings to full_df.
    """
    # Background sample for SHAP
    background = (
        X_train.sample(n=min(2000, len(X_train)), random_state=42)
        .astype("float64")
    )

    def model_predict_proba(X_input):
        return model.predict_proba(X_input)[:, 1]

    explainer = shap.Explainer(model_predict_proba, background)

    # ---- Global SHAP: on a validation sample ----
    sample_X = (
        X_val.sample(n=min(4000, len(X_val)), random_state=42)
        .astype("float64")
    )
    shap_result = explainer(sample_X)
    shap_values = shap_result.values  # (n_samples, n_features)
    print("SHAP matrix shape (val sample):", shap_values.shape)

    shap_global = (
        pd.DataFrame(
            {
                "feature": sample_X.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("Top 10 SHAP features:\n", shap_global.head(10))

    # ---- Local SHAP: for all rows we want to enrich ----
    df_for_shap = full_df[CANDIDATE_FEATURES].astype("float64")
    shap_full = explainer(df_for_shap).values  # (N, n_features)

    def top_local_features(shap_row, n=4):
        abs_vals = np.abs(shap_row)
        idx = np.argsort(-abs_vals)[:n]
        return [CANDIDATE_FEATURES[i] for i in idx]

    full_df = full_df.copy()
    full_df["top_shap_features"] = [
        top_local_features(shap_full[i], n=4) for i in range(len(shap_full))
    ]

    # Human-readable SHAP summary sentence with sign + magnitude
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
        row_shap_summary(shap_full[i]) for i in range(len(shap_full))
    ]

    # ------------------------------------------------------------------
    # 3. RISK SCORE (MODEL-BASED, PRODUCTION READY)
    # ------------------------------------------------------------------
    probs = model.predict_proba(df_for_shap)[:, 1]

    # Base risk score: 0–100
    full_df["risk_score_prob"] = probs
    full_df["risk_score"] = (probs * 100).round(2)

    # Optional conservative uplift if rule-based anomaly triggered
    if "is_anomaly_label" in full_df.columns:
        full_df.loc[
            full_df["is_anomaly_label"] == 1, "risk_score"
        ] = (
            full_df.loc[full_df["is_anomaly_label"] == 1, "risk_score"] + 8
        ).clip(upper=100)

    # ------------------------------------------------------------------
    # 4. RECOMMENDED ACTION (BUSINESS LAYER)
    # ------------------------------------------------------------------
    def recommend_action(score, shap_text):
        shap_text = str(shap_text).lower()

        if score >= 80:
            return "BLOCK & INVESTIGATE"

        if score >= 60:
            return "HOLD & MANUAL REVIEW"

        if score >= 40:
            # escalate if amount / velocity / geo signals present
            if any(k in shap_text for k in ["amount", "txn_cnt", "country", "city"]):
                return "REVIEW - PRIORITY (behavioral spike)"
            return "REVIEW / MONITOR"

        return "MONITOR / NO ACTION"

    full_df["recommended_action"] = full_df.apply(
        lambda r: recommend_action(r["risk_score"], r["shap_summary"]),
        axis=1,
    )

    return shap_global, full_df


def run_xgb_and_shap():
    """
    Orchestrator:
    - Load feature-engineered data
    - Train XGB
    - Compute SHAP
    - Save: XGB model, SHAP global importance, SHAP-enriched dataset.
    """
    ensure_dir(MODELS_DIR)

    # read FE dataset
    fe_path = processed_shap_path()
    df = pd.read_csv(fe_path,low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"Loaded feature-engineered data from {fe_path}, shape {df.shape}")

    model, X_train, y_train, X_val, y_val = train_xgb_classifier(df)
    shap_global, df_with_shap = compute_shap_for_model(model, X_train, X_val, df)

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgb_anomaly_model.json")
    model.save_model(model_path)
    print(f"Saved XGBoost model to {model_path}")

    # Save global SHAP
    shap_global_path = os.path.join(MODELS_DIR, "shap_global_top20.csv")
    shap_global.head(20).to_csv(shap_global_path, index=False)
    print(f"Saved SHAP global importance to {shap_global_path}")

    # Save SHAP-enriched dataframe for downstream SLM
    shap_path = processed_shap_path()
    df_with_shap.to_csv(shap_path, index=False)
    print(f"Saved SHAP-enriched dataset to {shap_path}, shape {df_with_shap.shape}")

    return df_with_shap, shap_global


if __name__ == "__main__":
    run_xgb_and_shap()
