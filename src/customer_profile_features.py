#!/usr/bin/env python
# coding: utf-8

import hashlib
import numpy as np
import pandas as pd

from .utils import processed_shap_path, DATA_PROCESSED_DIR, ensure_dir


def stable_hash(val):
    return int(hashlib.md5(str(val).encode()).hexdigest(), 16)


def run_customer_profile_features() -> pd.DataFrame:
    df = pd.read_csv(processed_shap_path(),low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

    # -----------------------------
    # Customer profile aggregation
    # -----------------------------
    cust_profile = (
        df.groupby("client_id")
        .agg(
            cust_spend_min=("txn_amount", "min"),
            cust_spend_max=("txn_amount", "max"),
            cust_avg_hour=("txn_hour", "mean"),
            cust_txn_hour_std=("txn_hour", "std"),
            cust_top_city=("txn_city", lambda x: x.mode().iat[0]),
            cust_top_country=("txn_country", lambda x: x.mode().iat[0]),
            cust_top_mcc=("mcc", lambda x: x.mode().iat[0]),
            cust_weekend_bias=(
                "txn_date",
                lambda x: x.dt.dayofweek.isin([5, 6]).mean(),
            ),
        )
        .reset_index()
    )

    cust_profile["cust_txn_hour_std"] = cust_profile["cust_txn_hour_std"].fillna(1)

    # -----------------------------
    # Deterministic synthetic fields
    # -----------------------------
    cust_profile["customer_tenure_w_bank"] = cust_profile["client_id"].apply(
        lambda x: 6 + stable_hash(x) % 120
    )
    cust_profile["cust_past_disputes_cnt"] = cust_profile["client_id"].apply(
        lambda x: stable_hash(x) % 10
    )
    cust_profile["cust_fraud_conf_cnt"] = (
        cust_profile["cust_past_disputes_cnt"] * 0.3
    ).astype(int)

    cust_profile["cust_past_fraud_exposure"] = np.where(
        cust_profile["cust_past_disputes_cnt"] > 0,
        cust_profile["cust_fraud_conf_cnt"]
        / cust_profile["cust_past_disputes_cnt"]
        * 100,
        0,
    )

    # -----------------------------
    # Merge back
    # -----------------------------
    df = df.merge(cust_profile, on="client_id", how="left")

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_shap_path(), index=False)

    print(f"Saved dataset WITH customer profiles → {processed_shap_path()} | {df.shape}")
    return df


if __name__ == "__main__":
    run_customer_profile_features()
