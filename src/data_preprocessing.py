#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from typing import Optional, List

from .utils import default_raw_path, ensure_dir, DATA_PROCESSED_DIR


def load_raw_transactions(path: Optional[str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the raw transactions CSV.
    """
    if path is None:
        path = default_raw_path()
    df = pd.read_csv(path, nrows=nrows)
    return df


def filter_sample_clients(df: pd.DataFrame, n_clients: int = 25) -> pd.DataFrame:
    """
    For development: keep data only for a subset of client_ids
    so we can iterate quickly without training on millions of rows.
    """
    if "client_id" not in df.columns:
        return df

    sample_clients = df["client_id"].dropna().astype(str).unique()[:n_clients]
    df_small = df[df["client_id"].astype(str).isin(sample_clients)].copy()
    return df_small


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: parse dates, sort, drop duplicates, remove obviously bad rows.
    Assumes columns similar to your Kaggle dataset.
    """
    df = df.copy()

    # date parsing
    if "txn_date" in df.columns:
        df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

    # drop duplicates by txn_id if present
    if "txn_id" in df.columns:
        df = df.drop_duplicates(subset=["txn_id"])

    # sort by client_id + date
    sort_cols = [c for c in ["client_id", "txn_date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # drop obvious nonsense, e.g., negative absurd amounts (you can adjust)
    if "txn_amount" in df.columns:
        df = df[df["txn_amount"].notna()]

    return df


def run_preprocessing(
    raw_path: Optional[str] = None,
    nrows: Optional[int] = None,
    n_clients_sample: Optional[int] = 50
) -> pd.DataFrame:
    """
    Orchestrator: load, filter, clean, and save a dev-sized dataset.
    """
    from .utils import processed_fe_path, ensure_dir

    df = load_raw_transactions(raw_path, nrows=nrows)
    if n_clients_sample:
        df = filter_sample_clients(df, n_clients_sample)
    df = basic_clean(df)

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_fe_path(), index=False)
    print(f"Saved cleaned subset to {processed_fe_path()} with shape {df.shape}")

    return df


if __name__ == "__main__":
    run_preprocessing()

