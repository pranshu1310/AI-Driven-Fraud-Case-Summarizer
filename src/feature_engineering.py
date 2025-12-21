#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from typing import List
import re
import hashlib
from sklearn.preprocessing import LabelEncoder

from .utils import processed_fe_path, processed_shap_path, ensure_dir, DATA_PROCESSED_DIR

# --- START: Enrichment function to paste into src/feature_engineering.py ---
def _stable_hash_int(val: str) -> int:
    """Stable hash -> integer using md5 so it's deterministic across runs/processes."""
    if pd.isna(val):
        val = "NA"
    b = str(val).encode("utf-8")
    return int(hashlib.md5(b).hexdigest(), 16)

def _deterministic_choice(client_id, options):
    idx = _stable_hash_int(client_id) % len(options)
    return options[idx]

def _deterministic_randint(client_id, low, high):
    rng = _stable_hash_int(client_id)
    return (rng % (high - low + 1)) + low

def _deterministic_float(client_id, low, high):
    rng = _stable_hash_int(client_id)
    scale = (rng % 10000) / 10000.0  # 0.0 - 0.9999
    return low + scale * (high - low)

def enrich_with_synthetic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add/normalize amounts, rename common raw columns into txn_*,
    add time-of-day categories, deterministic client summary, rolling counts,
    customer profile aggregation and engineered numeric/categorical encodings.
    """
    df = df.copy()

    # ---- 0) Ensure common column names exist (rename if raw names present) ----
    rename_map = {}
    if 'id' in df.columns and 'txn_id' not in df.columns:
        rename_map['id'] = 'txn_id'
    if 'date' in df.columns and 'txn_date' not in df.columns:
        rename_map['date'] = 'txn_date'
    if 'amount' in df.columns and 'txn_amount' not in df.columns:
        # we'll create txn_amount after cleaning
        pass
    if 'merchant_city' in df.columns and 'txn_city' not in df.columns:
        rename_map['merchant_city'] = 'txn_city'
    if 'merchant_state' in df.columns and 'txn_state' not in df.columns:
        rename_map['merchant_state'] = 'txn_state'
    if rename_map:
        df = df.rename(columns=rename_map)

    # ---- 1) Normalize Amount column (supports 'amount' or 'txn_amount') ----
    amount_col = None
    if 'txn_amount' in df.columns:
        amount_col = 'txn_amount'
    elif 'amount' in df.columns:
        amount_col = 'amount'

    if amount_col:
        raw = df[amount_col].astype(object).copy()
        df['amount_raw_for_norm'] = raw

        s = raw.astype(str).str.strip()
        s = s.str.replace('\xa0', '', regex=False)   # NBSP
        s = s.str.replace('−', '-', regex=False)     # unicode minus
        s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)  # (123) -> -123
        s = s.str.replace(r'[^\d\.-]', '', regex=True)        # remove currency signs, text, commas
        s = s.str.replace(r'\.(?=.*\.)', '', regex=True)      # keep only last dot
        s = s.str.replace(r'^(\d+)-$', r'-\1', regex=True)    # trailing minus -> leading
        # convert to numeric
        amt_numeric = pd.to_numeric(s, errors='coerce')
        # round and cast to nullable Int64 if you prefer integers; keep float if cents matter:
        # choose Int64 (as in your snippet)
        df['txn_amount'] = amt_numeric.round().astype('Int64')
        # diagnostics - store a few bad examples (non-blocking)
        bad_examples = df.loc[df['txn_amount'].isna() & df['amount_raw_for_norm'].notnull(), 'amount_raw_for_norm'].unique()[:20]
        if len(bad_examples) > 0:
            print("Example problematic amounts (first 20):", bad_examples)
    else:
        # If no amount column, create a placeholder so downstream code doesn't crash
        df['txn_amount'] = np.nan

    # ---- 2) Date/time extraction + hour category + day name ----
    if 'txn_date' in df.columns:
        df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')
    else:
        # try other common names
        for alt in ['date', 'trans_date', 'timestamp']:
            if alt in df.columns:
                df['txn_date'] = pd.to_datetime(df[alt], errors='coerce')
                break

    # categorize hour fn
    def categorize_hour(hour):
        if pd.isna(hour):
            return 'Unknown'
        hour = int(hour)
        if 4 <= hour < 8:
            return 'Early Morning'
        elif 8 <= hour < 11:
            return 'Late Morning'
        elif 11 <= hour < 14:
            return 'Early Noon'
        elif 14 <= hour < 17:
            return 'Late Noon'
        elif 17 <= hour < 19:
            return 'Early Evening'
        elif 19 <= hour < 21:
            return 'Late Evening'
        elif 21 <= hour < 24:
            return 'Early Night'
        else:
            return 'Late Night'

    df['txn_day'] = df['txn_date'].dt.date
    df['txn_hour'] = df['txn_date'].dt.hour
    df['txn_time_category'] = df['txn_hour'].apply(lambda x: categorize_hour(x) if not pd.isna(x) else 'Unknown')
    df['txn_day_of_week'] = df['txn_date'].dt.day_name()

    # ---- 3) Map txn_state -> txn_country for US states + fill others ----
    import numpy as _np
    us_states = {
        'ND': 'United States', 'IA': 'United States', 'CA': 'United States', 'IN': 'United States',
        'MD': 'United States', 'NY': 'United States', 'TX': 'United States', 'HI': 'United States',
        'PA': 'United States', 'WI': 'United States', 'GA': 'United States', 'AL': 'United States',
        'CT': 'United States', 'WA': 'United States', 'MA': 'United States', 'CO': 'United States',
        'NJ': 'United States', 'OK': 'United States', 'MT': 'United States', 'FL': 'United States',
        'AZ': 'United States', 'KY': 'United States', 'LA': 'United States', 'IL': 'United States',
        'OH': 'United States', 'MO': 'United States', 'MI': 'United States', 'KS': 'United States',
        'NC': 'United States', 'AR': 'United States', 'TN': 'United States', 'NM': 'United States',
        'SC': 'United States', 'MN': 'United States', 'NV': 'United States', 'OR': 'United States',
        'VA': 'United States', 'SD': 'United States', 'WV': 'United States', 'ME': 'United States',
        'MS': 'United States', 'RI': 'United States', 'NH': 'United States', 'DE': 'United States',
        'VT': 'United States', 'ID': 'United States', 'NE': 'United States', 'DC': 'United States',
        'UT': 'United States', 'WY': 'United States', 'AK': 'United States', 'AA': 'United States'
    }
    if 'txn_state' in df.columns:
        df['txn_country'] = df['txn_state'].map(us_states)
        df['txn_country'] = df['txn_country'].fillna(df['txn_state'])
    else:
        # if no txn_state, try merchant_country or country
        if 'merchant_country' in df.columns:
            df['txn_country'] = df['merchant_country'].fillna('Unknown')
        elif 'country' in df.columns:
            df['txn_country'] = df['country'].fillna('Unknown')
        else:
            df['txn_country'] = 'Unknown'
    df['txn_country'] = df['txn_country'].replace({_np.nan: 'Unknown'})

    # ---- 4) Deterministic client-level synthetic attributes ----
    # unique countries list
    countries = df['txn_country'].dropna().unique().tolist() or ['Unknown']
    income_segments = ['Low', 'Medium', 'High']
    typical_hours = ['Early Morning', 'Late Morning', 'Early Noon', 'Late Noon',
                     'Early Evening', 'Late Evening', 'Early Night', 'Late Night']
    travel_flags = [0, 1]

    if 'client_id' in df.columns:
        # base client_summary frame
        client_df = df[['client_id']].drop_duplicates().copy()
        client_df['cust_home_country'] = client_df['client_id'].apply(lambda x: _deterministic_choice(x, countries))
        client_df['cust_income_segment'] = client_df['client_id'].apply(lambda x: _deterministic_choice(x, income_segments))
        client_df['cust_typical_hours'] = client_df['client_id'].apply(lambda x: _deterministic_choice(x, typical_hours))
        client_df['cust_travel_flag'] = client_df['client_id'].apply(lambda x: _deterministic_choice(x, travel_flags))
        client_df['customer_tenure_w_bank'] = client_df['client_id'].apply(lambda x: _deterministic_randint(x, 6, 120))
        client_df['cust_past_disputes_cnt'] = client_df['client_id'].apply(lambda x: _deterministic_randint(x, 0, 10))
        # cust_fraud_conf_cnt depends on cust_past_disputes_cnt; compute robustly
        client_df['cust_fraud_conf_cnt'] = client_df.apply(
            lambda r: int(r['cust_past_disputes_cnt'] * _deterministic_float(r['client_id'], 0.1, 0.6))
            if r['cust_past_disputes_cnt'] > 0 else 0,
            axis=1
        )
        client_df['cust_past_fraud_exposure'] = client_df.apply(
            lambda r: (r['cust_fraud_conf_cnt'] / r['cust_past_disputes_cnt'] * 100)
            if r['cust_past_disputes_cnt'] > 0 else 0,
            axis=1
        )
        # Merge to transactions
        df = df.merge(client_df, on='client_id', how='left')

    # ---- 5) Cust-level averages and historical monthly spend ----
    if 'client_id' in df.columns and 'txn_amount' in df.columns:
        agg = df.groupby('client_id').agg(
            cust_avg_txn_amt=('txn_amount', 'mean'),
            cust_txn_stddev=('txn_amount', 'std'),
        ).reset_index()
        agg['cust_txn_stddev'] = agg['cust_txn_stddev'].fillna(1)
        # transactions per client
        txn_counts = df.groupby('client_id')['txn_id'].count() if 'txn_id' in df.columns else df.groupby('client_id').size()
        months_in_data = 1.0
        if df['txn_date'].notna().any():
            # avoid zero months
            min_date = df['txn_date'].min()
            max_date = df['txn_date'].max()
            months_in_data = max(1.0, ((max_date - min_date).days / 30.0))
        agg['cust_historical_monthly_spend'] = agg['cust_avg_txn_amt'] * (txn_counts.values / months_in_data)
        df = df.merge(agg, on='client_id', how='left')

    # ---- 6) Rolling transaction counts in given minute-windows ----
    def rolling_txn_counts_local(df_input, windows=[30, 60, 90, 1440]):
        # expects txn_date present and client_id present
        out_rows = []
        if 'client_id' not in df_input.columns or 'txn_date' not in df_input.columns:
            return pd.DataFrame(columns=['client_id', 'txn_date'])
        for cid, grp in df_input.groupby('client_id', sort=False):
            grp = grp.sort_values('txn_date').reset_index(drop=True)
            times = grp['txn_date'].astype('datetime64[ns]').values
            out = grp[['client_id', 'txn_date']].copy()
            for w in windows:
                window_ns = np.timedelta64(w, 'm')
                idx = np.searchsorted(times, times - window_ns, side='left')
                counts = np.arange(len(times)) - idx
                out[f'txn_cnt_inlast{w}mins'] = counts
            out_rows.append(out)
        if out_rows:
            return pd.concat(out_rows, ignore_index=True)
        else:
            return pd.DataFrame(columns=['client_id', 'txn_date'])

    if 'client_id' in df.columns and 'txn_date' in df.columns:
        txn_counts_df = rolling_txn_counts_local(df[['client_id', 'txn_date']])
        # merge counts
        df = df.merge(txn_counts_df, on=['client_id', 'txn_date'], how='left')

    # ---- 7) Client behavioural profile aggregates ----
    if 'client_id' in df.columns:
        cust_profile = (
            df.groupby("client_id").agg(
                cust_avg_hour=("txn_hour", "mean"),
                cust_txn_hour_std=("txn_hour", "std"),
                cust_spend_min=("txn_amount", "min"),
                cust_spend_max=("txn_amount", "max"),
                cust_top_city=("txn_city", lambda x: x.mode().iat[0] if not x.mode().empty else np.nan),
                cust_top_country=("txn_country", lambda x: x.mode().iat[0] if not x.mode().empty else np.nan),
                cust_top_mcc=("mcc", lambda x: x.mode().iat[0] if not x.mode().empty else np.nan),
                cust_daily_txn_avg=("txn_id", lambda x: len(x) / x.nunique() if x.nunique() > 0 else 0),
                cust_weekend_bias=("txn_day_of_week", lambda x: (x.isin(["Saturday", "Sunday"]).mean())),
            )
            .reset_index()
        )
        df = df.merge(cust_profile, on='client_id', how='left')
        df['cust_txn_hour_std'] = df['cust_txn_hour_std'].fillna(1)

    # ---- 8) Derived features and flags ----
    df['txn_amount_ratio_to_avg'] = df['txn_amount'] / (df['cust_avg_txn_amt'] + 1e-9)
    df['txn_amount_minus_avg'] = df['txn_amount'] - df['cust_avg_txn_amt']
    df['txn_hour_sin'] = np.sin(2 * np.pi * (df['txn_hour'].fillna(0) / 24.0))
    df['txn_hour_cos'] = np.cos(2 * np.pi * (df['txn_hour'].fillna(0) / 24.0))
    df['cust_avg_hour_sin'] = np.sin(2 * np.pi * (df['cust_avg_hour'].fillna(0) / 24.0))
    df['cust_avg_hour_cos'] = np.cos(2 * np.pi * (df['cust_avg_hour'].fillna(0) / 24.0))

    df['is_high_vs_customer_max'] = (df['txn_amount'] > df['cust_spend_max']).astype(int)
    df['is_above_historical_monthly_avg'] = (df['txn_amount'] > (df['cust_historical_monthly_spend'] / 10)).astype(int)

    # velocity ratios (guard divide by zero)
    df['txn_cnt_inlast30mins'] = df.get('txn_cnt_inlast30mins', 0)
    df['txn_cnt_inlast60mins'] = df.get('txn_cnt_inlast60mins', 0)
    df['txn_cnt_inlast90mins'] = df.get('txn_cnt_inlast90mins', 0)
    df['txn_cnt_inlast1440mins'] = df.get('txn_cnt_inlast1440mins', 0)

    df['txn_cnt_ratio_60_30'] = ((df['txn_cnt_inlast60mins'] + 1) / (df['txn_cnt_inlast30mins'] + 1)).astype(float)
    df['txn_cnt_ratio_90_60'] = ((df['txn_cnt_inlast90mins'] + 1) / (df['txn_cnt_inlast60mins'] + 1)).astype(float)

    # recency
    df = df.sort_values(['client_id', 'txn_date'])
    if 'client_id' in df.columns and 'txn_date' in df.columns:
        df['prev_txn_date'] = df.groupby('client_id')['txn_date'].shift(1)
        df['mins_since_prev_txn'] = (df['txn_date'] - df['prev_txn_date']).dt.total_seconds() / 60.0
        df['mins_since_prev_txn'] = df['mins_since_prev_txn'].fillna(999999)

    # ---- 9) Lightweight categorical encodings for mcc / city / country ----
    for col in ['mcc', 'txn_city', 'txn_country', 'cust_top_mcc', 'cust_top_city']:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('NA')
        else:
            df[col] = 'NA'

    le_mcc = LabelEncoder()
    df['mcc_enc'] = le_mcc.fit_transform(df['mcc'])

    le_city = LabelEncoder()
    df['city_enc'] = le_city.fit_transform(df['txn_city'])

    le_country = LabelEncoder()
    df['country_enc'] = le_country.fit_transform(df['txn_country'])

    # final sanity: drop temporary amount_raw_for_norm if you don't want it in outputs
    # df = df.drop(columns=['amount_raw_for_norm'], errors='ignore')
    return df



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

    df = enrich_with_synthetic_columns(df)
    
    df = ensure_candidate_features(df)

    ensure_dir(DATA_PROCESSED_DIR)
    df.to_csv(processed_shap_path(), index=False)
    print(f"Saved feature-engineered dataset to {processed_shap_path()} with shape {df.shape}")
    return df


if __name__ == "__main__":
    run_feature_engineering()

