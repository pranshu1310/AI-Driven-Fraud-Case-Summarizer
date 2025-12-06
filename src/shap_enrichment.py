#!/usr/bin/env python
# coding: utf-8

import ast
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from .utils import (
    processed_shap_path,
    slm_training_data_path,
    ensure_dir,
    DATA_PROCESSED_DIR,
)


# -------------------------------------------------------------------
# 1) Parse SHAP info into list of (feature, impact)
# -------------------------------------------------------------------
def parse_top_shap(row: pd.Series) -> List[Tuple[str, float]]:
    """
    Returns list of (feature, impact) pairs.

    Priority:
    1. Parse 'shap_summary' if present (keeps +/- SHAP values).
       Example: "Top factors: feat1 +0.049, feat2 -0.012, ..."
    2. Fallback to 'top_shap_features' (no numeric impact) -> impact=None.
    """
    # 1) Try shap_summary first
    summary = row.get("shap_summary", "")
    if isinstance(summary, str) and "Top factors" in summary:
        parts = re.findall(r"([A-Za-z0-9_]+)\s*([+-]?\d+\.\d+)", summary)
        if parts:
            return [(feat, float(val)) for feat, val in parts]

    # 2) Fallback to top_shap_features
    raw = row.get("top_shap_features", None)

    # if already list of strings
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return [(x, None) for x in raw]

    # stringified list: "['city_enc','mcc_enc']"
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [(str(x), None) for x in parsed]
        except Exception:
            pass

    # 3) Nothing usable
    return []


# -------------------------------------------------------------------
# 2) Feature -> natural language using sign (your custom logic)
# -------------------------------------------------------------------
def describe_feature_with_sign(feat: str, impact):
    """
    Turn (feature, impact) into a natural phrase.
    Uses sign when it makes sense. impact can be None.
    """
    f = feat.lower()
    pos = impact is not None and impact > 0
    neg = impact is not None and impact < 0

    # Amount-like
    if "amount" in f:
        if impact is None:
            return "transaction amount deviating from the customer’s usual spend"
        if pos:
            return "a higher-than-usual transaction amount versus the customer’s history"
        if neg:
            return "a lower-than-usual transaction amount compared with typical behaviour"

    # Count / frequency
    if "cnt" in f or "freq" in f:
        if impact is None:
            return "an unusual pattern in transaction frequency"
        if pos:
            return "a spike in recent transaction count relative to normal activity"
        if neg:
            return "a drop in recent transaction volume compared to prior patterns"

    # Time gap in minutes
    if "mins_since_prev_txn" in f or "mins" in f:
        if impact is None:
            return "irregular spacing between consecutive transactions"
        if pos:
            return "a very short gap between this and the previous transaction"
        if neg:
            return "an unusually long gap between this and the previous transaction"

    # Time-of-day
    if "hour" in f or "time" in f:
        return "the transaction occurring at an unusual time of day for this customer"

    # Location
    if "city" in f:
        return "a change in city or location compared with the customer’s typical profile"
    if "country" in f:
        return "activity originating from an unexpected country for this customer"

    # Merchant category
    if "mcc" in f or "merchant" in f:
        return "spend at a merchant or category that is not typical for this customer"

    # Default fallback
    return f"behaviour linked to '{feat.replace('_',' ')}' that deviates from normal patterns"


def shap_to_reasons(pairs: List[Tuple[str, float]], max_reasons: int = 3) -> List[str]:
    """
    Convert list of (feat, impact) into up to max_reasons natural-language reasons.

    - Prefer features with positive SHAP (risk-raising).
    - If no positive values, fall back to largest |impact|.
    - If all impacts are None, just use features.
    """
    if not isinstance(pairs, list) or len(pairs) == 0:
        return []

    numeric_pairs = [(f, v) for f, v in pairs if isinstance(v, (float, int))]
    none_pairs = [(f, v) for f, v in pairs if v is None]

    # Prefer positive SHAP
    pos_pairs = [(f, v) for f, v in numeric_pairs if v > 0]
    if pos_pairs:
        candidates = sorted(pos_pairs, key=lambda x: -abs(x[1]))
    elif numeric_pairs:
        # No positive -> use largest magnitude
        candidates = sorted(numeric_pairs, key=lambda x: -abs(x[1]))
    else:
        candidates = none_pairs

    chosen = candidates[:max_reasons]
    reasons = [describe_feature_with_sign(f, v) for f, v in chosen]
    return reasons


# -------------------------------------------------------------------
# 3) 20 templated explanations (silver labels)
# -------------------------------------------------------------------
def build_explanation_from_reasons(reasons: List[str]) -> str:
    """
    Use up to 20 different templates to generate a natural explanation
    from a list of short reason phrases.
    """
    if not reasons:
        return "No material anomaly indicators were detected for this transaction."

    r1 = reasons[0]
    r2 = reasons[1] if len(reasons) > 1 else None
    r3 = reasons[2] if len(reasons) > 2 else None  # currently unused but available

    joined_all = ", ".join(reasons)
    joined_two = ", and ".join(reasons[:2]) if len(reasons) >= 2 else r1

    templates = [
        lambda: f"The transaction appears anomalous primarily due to {r1}.",
        lambda: f"The alert is driven by {joined_two}, which deviates from the customer’s normal behaviour.",
        lambda: f"Risk is elevated because the system detected {joined_all}.",
        lambda: f"This transaction stands out versus historical patterns, especially due to {r1}.",
        lambda: f"Anomaly drivers include {joined_all}, indicating behaviour outside expected norms.",
        lambda: f"The case was flagged as the model identified {joined_two} as unusual for this customer.",
        lambda: f"Model explanations highlight {joined_all} as key signals of potential fraud.",
        lambda: f"The transaction triggers concern due to {r1}, supported by additional irregularities in recent activity.",
        lambda: f"From an analytical standpoint, {joined_two} are the main factors contributing to the alert.",
        lambda: f"The model assigns higher risk because it observes {joined_all} compared with prior behaviour.",
        lambda: f"Behaviour on this transaction diverges from the customer’s profile, notably through {r1}.",
        lambda: f"The anomaly score is influenced by {joined_two}, which are not consistent with historical trends.",
        lambda: f"Several patterns, including {joined_all}, differentiate this transaction from typical activity.",
        lambda: f"Alert generation is largely explained by {r1}, with additional contribution from other secondary signals.",
        lambda: f"The system highlights {joined_two} as the strongest indicators of risk relative to this customer’s baseline.",
        lambda: f"Compared to usual behaviour, this transaction shows {joined_all}, justifying closer review.",
        lambda: f"The model explanation points to {r1} as the primary driver, with further deviation in related attributes.",
        lambda: f"Key features shaping the fraud decision include {joined_all}, all of which depart from normal patterns.",
        lambda: f"The transaction is flagged because {r1} represents an atypical pattern for this customer.",
        lambda: f"Overall, {joined_all} together create a risk pattern that warranted an anomaly alert.",
    ]

    # random choice is fine for training variety; if you want determinism, pick templates[0]
    import random

    template_fn = random.choice(templates)
    return template_fn()


# -------------------------------------------------------------------
# 4) Generative prompt builder (for SLM training & inference)
# -------------------------------------------------------------------
def build_generative_prompt_from_row(row: pd.Series) -> str:
    """
    Build the generative prompt for the SLM based on transaction metadata
    and parsed SHAP pairs (_top_shap_pairs).
    """
    pairs = row.get("_top_shap_pairs", [])
    if isinstance(pairs, list):
        shap_text = ", ".join(
            f"{feat.replace('_', ' ')} ({impact:+.3f})"
            if isinstance(impact, (float, int))
            else feat.replace("_", " ")
            for feat, impact in pairs[:5]
        )
    else:
        shap_text = "not available"

    prompt = (
        "You are a fraud investigator assistant. Generate a concise 1–2 line explanation "
        "for why the transaction may be anomalous. Use professional language suitable for case notes.\n\n"
        "Transaction Details:\n"
        f"- Amount: ${row.get('txn_amount')}\n"
        f"- Location: {row.get('txn_city')}, {row.get('txn_country')}\n"
        f"- Minutes since previous transaction: {row.get('mins_since_prev_txn')}\n"
        f"- Transactions in last 24 hours: {row.get('txn_cnt_inlast1440mins')}\n\n"
        "Model Explanation Signals (SHAP-derived):\n"
        f"{shap_text}\n\n"
        "Instruction: Based on the model signals above, describe in your own words what looks unusual "
        "about this transaction, in 1–2 sentences, as if writing a case note for fraud review."
    )
    return prompt


# -------------------------------------------------------------------
# 5) Build SLM dataset from SHAP-enriched dataframe
# -------------------------------------------------------------------
def build_slm_dataset() -> pd.DataFrame:
    """
    Load SHAP-enriched dataframe (with top_shap_features + shap_summary),
    and build:

      - _top_shap_pairs  : list[(feature, impact)]
      - _shap_reasons    : list[str] natural-language reasons
      - target_text      : a single explanation from 20 templates
      - input_text       : generative prompt

    Saves the full SLM dataset (with all original columns) to slm_training_data.csv.
    """
    shap_path = processed_shap_path()
    df = pd.read_csv(shap_path)
    print(f"Loaded SHAP-enriched data from {shap_path}, shape {df.shape}")

    # 1) Parse SHAP into structured pairs
    df["_top_shap_pairs"] = df.apply(parse_top_shap, axis=1)

    # 2) Turn SHAP into natural reasons
    df["_shap_reasons"] = df["_top_shap_pairs"].apply(shap_to_reasons)

    # 3) Silver-label target_text (explanation) using the 20 templates
    df["target_text"] = df["_shap_reasons"].apply(build_explanation_from_reasons)

    # 4) Build input_text prompts
    df["input_text"] = df.apply(build_generative_prompt_from_row, axis=1)

    # Save training data
    ensure_dir(DATA_PROCESSED_DIR)
    out_path = slm_training_data_path()
    df.to_csv(out_path, index=False)
    print(f"SLM training dataset saved to {out_path}, shape {df.shape}")

    return df


if __name__ == "__main__":
    build_slm_dataset()
