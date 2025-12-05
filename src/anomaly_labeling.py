#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from typing import Optional

from .utils import processed_shap_path
from .feature_engineering import CANDIDATE_FEATURES


def create_rule_based_label(df: pd.DataFrame) -> pd.Series:
    """
    Simple rule-based label as a fallback:
    e.g., large amount + high ratio + recent bursts.
    Adjust thresholds as per your business logic.
    """
    df = df.copy()
    cond_amount = df["txn_amount"] > (df["cust_spend_max"] * 1.2)
    cond_ratio = df["txn_amount_ratio_to_avg"] > 2.0
    cond_burst = df["txn_cnt_inlast60mins"] > 3

    label = (cond_amount | cond_ratio | cond_burst).astype(int)
    return label


def run_anomaly_labeling(path: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure we have a binary label column is_anomaly_label.
    If it already exists, we keep it; otherwise we generate it.
    """
    if path is None:
        path = processed_shap_path()
    df = pd.read_csv(path)

    if "is_anomaly_label" not in df.columns:
        df["is_anomaly_label"] = create_rule_based_label(df)
        print("Created is_anomaly_label using rule-based heuristics.")
    else:
        print("Reusing existing is_anomaly_label column.")

    df.to_csv(path, index=False)
    print(f"Updated dataset with labels saved to {path}, shape {df.shape}")
    return df


if __name__ == "__main__":
    run_anomaly_labeling()

