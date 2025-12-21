#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from .utils import processed_shap_path, DATA_PROCESSED_DIR, ensure_dir


def run_velocity_features() -> pd.DataFrame:
    """
    Batch-only velocity features.
    Memory-safe: computes in-place, no large intermediate DataFrames.
    """
    df = pd.read_csv(processed_shap_path(), low_memory=False)

    # Ensure datetime
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

    # Sort once (critical)
    df = df.sort_values(["client_id", "txn_date"]).reset_index(drop=True)

    # Pre-allocate columns (IMPORTANT for memory)
    df["prev_txn_date"] = pd.NaT
    df["mins_since_prev_txn"] = np.nan
    df["txn_cnt_inlast30mins"] = 0
    df["txn_cnt_inlast60mins"] = 0
    df["txn_cnt_inlast1440mins"] = 0
    df["txn_cnt_ratio_60_30"] = 0.0

    # --------------------------------------------------
    # Compute per-client velocity in-place
    # --------------------------------------------------
    for cid, idx in df.groupby("client_id").groups.items():
        idx = np.array(idx)
        times = df.loc[idx, "txn_date"].values.astype("datetime64[ns]")

        # --- Recency ---
        prev_times = np.roll(times, 1)
        prev_times[0] = np.datetime64("NaT")
        df.loc[idx, "prev_txn_date"] = prev_times

        mins = (times - prev_times) / np.timedelta64(1, "m")
        mins[0] = 999999
        df.loc[idx, "mins_since_prev_txn"] = mins

        # --- Rolling counts ---
        for window, col in [
            (30, "txn_cnt_inlast30mins"),
            (60, "txn_cnt_inlast60mins"),
            (1440, "txn_cnt_inlast1440mins"),
        ]:
            delta = np.timedelta64(window, "m")
            left_idx = np.searchsorted(times, times - delta, side="left")
            counts = np.arange(len(times)) - left_idx
            df.loc[idx, col] = counts

    # --------------------------------------------------
    # Velocity ratio
    # --------------------------------------------------
    df["txn_cnt_ratio_60_30"] = (
        (df["txn_cnt_inlast60mins"] + 1)
        / (df["txn_cnt_inlast30mins"] + 1)
    )

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_shap_path(), index=False)

    print(
        f"Saved dataset WITH velocity features → {processed_shap_path()} | shape={df.shape}"
    )
    return df


if __name__ == "__main__":
    run_velocity_features()
