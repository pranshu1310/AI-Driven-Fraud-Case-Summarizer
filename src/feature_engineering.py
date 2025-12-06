#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from typing import List

from .utils import processed_fe_path, processed_shap_path, ensure_dir, DATA_PROCESSED_DIR


# Candidate features, aligned with what you used for XGBoost
CANDIDATE_FEATURES: List[str] = [
    "txn_amount", "txn_amount_ratio_to_avg", "txn_amount_minus_avg",
    "cust_avg_txn_amt", "cust_txn_stddev", "cust_spend_min", "cust_spend_max",
    "cust_historical_monthly_spend", "cust_avg_hour", "cust_txn_hour_std",
    "txn_hour", "txn_hour_sin", "txn_hour_cos", "cust_avg_hour_sin", "cust_avg_hour_cos",
    "txn_cnt_inlast30mins", "txn_cnt_inlast60mins", "txn_cnt_inlast90mins", "txn_cnt_inlast1440mins",
    "txn_cnt_ratio_60_30", "txn_cnt_ratio_90_60", "mins_since_prev_txn",
    "review_time_left", "cust_weekend_bias",
    "mcc_enc", "city_enc", "country_enc",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure txn_date is datetime. Then safely compute hour, day, and sinusoidal encodings.
    Never breaks even if txn_date is missing or unparseable.
    """
    df = df.copy()

    # 1) Make txn_date a datetime if it exists
    if "txn_date" in df.columns:
        df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

        if df["txn_date"].isna().all():
            print("⚠️ WARNING: txn_date exists but could not be parsed to datetime — skipping time features.")
            return df
    else:
        print("⚠️ WARNING: txn_date column not found — skipping time features.")
        return df

    # 2) Now safely add hour/day
    df["txn_hour"] = df["txn_date"].dt.hour
    df["txn_day"]  = df["txn_date"].dt.day

    # 3) Add sin/cos encoding
    df["txn_hour_sin"] = np.sin(2 * np.pi * df["txn_hour"] / 24.0)
    df["txn_hour_cos"] = np.cos(2 * np.pi * df["txn_hour"] / 24.0)

    return df



def ensure_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure all CANDIDATE_FEATURES exist. If your dataset already
    has them (from your earlier pipeline), this will mostly be a no-op.
    """
    df = df.copy()
    df = add_time_features(df)

    # You can add guards here if some columns are missing
    missing_cols = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing_cols:
        print("WARNING: These candidate features are missing and will be filled with 0:", missing_cols)
        for c in missing_cols:
            df[c] = 0.0

    return df


def run_feature_engineering() -> pd.DataFrame:
    """
    Load cleaned data, ensure engineered features, and save.
    """
    df = pd.read_csv(processed_fe_path())
    df = ensure_candidate_features(df)

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_shap_path(), index=False)
    print(f"Saved feature-engineered dataset to {processed_shap_path()} with shape {df.shape}")
    return df


if __name__ == "__main__":
    run_feature_engineering()

