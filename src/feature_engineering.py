#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder

from .utils import processed_fe_path, processed_shap_path, ensure_dir, DATA_PROCESSED_DIR


# --------------------------------------------------
# XGB candidate features (UNCHANGED)
# --------------------------------------------------
CANDIDATE_FEATURES: List[str] = [
    "txn_amount",
    "txn_hour",
    "txn_hour_sin",
    "txn_hour_cos",
    "cust_avg_txn_amt",
    "cust_txn_stddev",
    "mcc_enc",
    "city_enc",
    "country_enc",
]


def run_feature_engineering() -> pd.DataFrame:
    # ---- Load safely (avoid dtype warning explosion) ----
    df = pd.read_csv(processed_fe_path(), low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # --------------------------------------------------
    # 1. Amount normalization (defensive)
    # --------------------------------------------------
    df["txn_amount"] = (
        df["txn_amount"]
        .astype(str)
        .str.replace(r"[^\d\.-]", "", regex=True)
    )
    df["txn_amount"] = pd.to_numeric(df["txn_amount"], errors="coerce")

    # --------------------------------------------------
    # 2. Time features (SAFE)
    # --------------------------------------------------
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

    df["txn_hour"] = df["txn_date"].dt.hour
    df["txn_hour"] = df["txn_hour"].fillna(0).astype(int)

    df["txn_hour_sin"] = np.sin(2 * np.pi * df["txn_hour"] / 24.0)
    df["txn_hour_cos"] = np.cos(2 * np.pi * df["txn_hour"] / 24.0)

    # --------------------------------------------------
    # 3. Ensure txn_country EXISTS (ROOT CAUSE FIX)
    # --------------------------------------------------
    us_states = {
        "CA": "United States",
        "NY": "United States",
        "TX": "United States",
        "FL": "United States",
        "IL": "United States",
    }

    if "txn_country" not in df.columns:
        if "txn_state" in df.columns:
            df["txn_country"] = df["txn_state"].map(us_states)
            df["txn_country"] = df["txn_country"].fillna("Unknown")
        else:
            df["txn_country"] = "Unknown"

    # --------------------------------------------------
    # 4. Light customer aggregates (SAFE, REQUIRED)
    # --------------------------------------------------
    if "client_id" in df.columns:
        cust = (
            df.groupby("client_id")
            .agg(
                cust_avg_txn_amt=("txn_amount", "mean"),
                cust_txn_stddev=("txn_amount", "std"),
            )
            .reset_index()
        )
        cust["cust_txn_stddev"] = cust["cust_txn_stddev"].fillna(1)
        df = df.merge(cust, on="client_id", how="left")
    else:
        df["cust_avg_txn_amt"] = 0.0
        df["cust_txn_stddev"] = 1.0

    # --------------------------------------------------
    # 5. Encoding (DEFENSIVE)
    # --------------------------------------------------
    for col in ["mcc", "txn_city", "txn_country"]:
        if col not in df.columns:
            df[col] = "NA"
        df[col] = df[col].astype(str).fillna("NA")

    df["mcc_enc"] = LabelEncoder().fit_transform(df["mcc"])
    df["city_enc"] = LabelEncoder().fit_transform(df["txn_city"])
    df["country_enc"] = LabelEncoder().fit_transform(df["txn_country"])

    # --------------------------------------------------
    # 6. Final guard: ensure all model features exist
    # --------------------------------------------------
    for c in CANDIDATE_FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_shap_path(), index=False)

    print(
        f"Saved feature-engineered dataset → {processed_shap_path()} | shape={df.shape}"
    )
    return df


if __name__ == "__main__":
    run_feature_engineering()
