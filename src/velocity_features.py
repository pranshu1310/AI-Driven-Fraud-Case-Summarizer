#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from .utils import processed_shap_path, DATA_PROCESSED_DIR, ensure_dir


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch-only velocity features.
    Assumes:
      - txn_date is datetime
      - client_id exists
    """
    df = df.copy()
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

    # Sort for correct rolling behaviour
    df = df.sort_values(["client_id", "txn_date"])

    # -----------------------------
    # Recency feature
    # -----------------------------
    df["prev_txn_date"] = df.groupby("client_id")["txn_date"].shift(1)
    df["mins_since_prev_txn"] = (
        (df["txn_date"] - df["prev_txn_date"])
        .dt.total_seconds()
        .div(60)
    )
    df["mins_since_prev_txn"] = df["mins_since_prev_txn"].fillna(999999)

    # -----------------------------
    # Rolling counts (time-based)
    # -----------------------------
    def rolling_counts(grp: pd.DataFrame) -> pd.DataFrame:
        times = grp["txn_date"].values.astype("datetime64[ns]")
        out = grp[["txn_date"]].copy()

        for window, name in [
            (30, "txn_cnt_inlast30mins"),
            (60, "txn_cnt_inlast60mins"),
            (1440, "txn_cnt_inlast1440mins"),
        ]:
            delta = np.timedelta64(window, "m")
            idx = np.searchsorted(times, times - delta, side="left")
            out[name] = np.arange(len(times)) - idx

        return out

    velocity_parts = []
    for cid, grp in df.groupby("client_id", sort=False):
        part = rolling_counts(grp.reset_index(drop=True))
        part["client_id"] = cid
        velocity_parts.append(part)

    velocity_df = pd.concat(velocity_parts, ignore_index=True)

    # -----------------------------
    # Merge velocity back
    # -----------------------------
    df = df.merge(
        velocity_df,
        on=["client_id", "txn_date"],
        how="left",
    )

    # -----------------------------
    # Velocity ratios
    # -----------------------------
    df["txn_cnt_ratio_60_30"] = (
        (df["txn_cnt_inlast60mins"] + 1)
        / (df["txn_cnt_inlast30mins"] + 1)
    )

    return df


def run_velocity_features() -> pd.DataFrame:
    """
    Load SHAP-ready dataset → compute velocity → overwrite dataset.
    """
    df = pd.read_csv(processed_shap_path())
    df = compute_velocity_features(df)

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_shap_path(), index=False)

    print(
        f"Saved dataset WITH velocity features → {processed_shap_path()} | shape={df.shape}"
    )
    return df


if __name__ == "__main__":
    run_velocity_features()
