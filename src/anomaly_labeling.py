#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from typing import Optional

from .utils import processed_shap_path, ensure_dir, DATA_PROCESSED_DIR


# ------------------------------
# 1) Your rich anomaly rule logic
# ------------------------------
def is_anomaly_from_row(row):
    reasons = []

    # Amount vs historical boundaries
    if "cust_spend_max" in row and "cust_spend_min" in row:
        if row["txn_amount"] > row.get("cust_spend_max", np.inf):
            reasons.append("amount_exceeds_max")
        elif row["txn_amount"] < row.get("cust_spend_min", 0) * 0.7:
            reasons.append("unusually_small")

    # Hour deviation from average
    try:
        if abs(
            row["txn_hour"] - row.get("cust_avg_hour", row["txn_hour"])
        ) > (row.get("cust_txn_hour_std", 1.0) or 1.0) * 2:
            reasons.append("unusual_hour")
    except:
        pass

    # City mismatch
    if "cust_top_city" in row and row["txn_city"] != row.get("cust_top_city", row["txn_city"]):
        reasons.append("city_mismatch")

    # Country mismatch
    if "cust_top_country" in row and row["txn_country"] != row.get("cust_top_country", row["txn_country"]):
        reasons.append("country_mismatch")

    # Velocity
    if row.get("txn_cnt_inlast60mins", 0) > 5:
        reasons.append("high_velocity")

    return 1 if len(reasons) > 0 else 0


# ------------------------------
# 2) Wrapper to apply rule logic
# ------------------------------
def create_full_anomaly_labels(df: pd.DataFrame, noise_fraction: float = 0.30) -> pd.DataFrame:
    df = df.copy()

    print("Creating anomaly labels using full behavioral rule logic...")
    df["is_anomaly_label"] = df.apply(is_anomaly_from_row, axis=1)

    # Noise injection improves robustness
    mask = np.random.rand(len(df)) < noise_fraction
    df.loc[mask, "is_anomaly_label"] = 1 - df.loc[mask, "is_anomaly_label"]

    print(f"Injected noise into {mask.sum()} of {len(df)} rows ({noise_fraction*100:.1f}%).")
    print("Label distribution:", df["is_anomaly_label"].value_counts().to_dict())

    return df


# ------------------------------
# 3) Main entry point
# ------------------------------
def run_anomaly_labeling(input_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load feature-engineered dataset → create rule-based labels → save labeled version.
    """
    if input_path is None:
        input_path = processed_shap_path()

    df = pd.read_csv(input_path, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = create_full_anomaly_labels(df)

    ensure_dir(DATA_PROCESSED_DIR)
    out_path = processed_shap_path()
    df.to_csv(out_path, index=False)

    print(f"Saved labeled dataset to {out_path}, shape {df.shape}")
    return df


if __name__ == "__main__":
    run_anomaly_labeling()
