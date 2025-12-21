#!/usr/bin/env python
# coding: utf-8

import re
import ast
import random
from typing import List, Tuple

import pandas as pd

from .utils import (
    processed_shap_path,
    slm_training_data_path,
    ensure_dir,
    DATA_PROCESSED_DIR,
    set_seed,
)

# -------------------------------------------------------------------
# 1) Parse SHAP info into list of (feature, impact) pairs
#    - Prefer shap_summary (with numeric + / - impact)
#    - Fallback to top_shap_features (names only)
# -------------------------------------------------------------------

def parse_top_shap(row) -> List[Tuple[str, float]]:
    """
    Returns list of (feature, impact) pairs.

    Priority:
      1) shap_summary like:
         "Top factors: mins_since_prev_txn +0.049, city_enc -0.047, ..."
         → [("mins_since_prev_txn", 0.049), ("city_enc", -0.047), ...]
      2) top_shap_features column:
         - list of strings or stringified list
         → [("city_enc", None), ("mcc_enc", None), ...]
      3) No info → []
    """
    # --- 1) Try shap_summary first (keeps numeric sign) ---
    summary = row.get("shap_summary", "")
    if isinstance(summary, str) and "Top factors" in summary:
        parts = re.findall(r"([A-Za-z0-9_]+)\s*([+-]?\d+\.\d+)", summary)
        if parts:
            return [(feat, float(val)) for feat, val in parts]

    # --- 2) Fallback: top_shap_features (local feature names only) ---
    raw = row.get("top_shap_features", None)

    # already a python list of strings
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return [(x, None) for x in raw]

    # stringified list: "['city_enc', 'mcc_enc']"
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [(str(x), None) for x in parsed]
        except Exception:
            pass

    # --- 3) No usable info ---
    return []


# -------------------------------------------------------------------
# 2) Feature-group → natural language, with sign awareness
# -------------------------------------------------------------------

def describe_feature_with_sign(feat: str, impact):
    """
    Turn (feature, impact) into a short, human phrase.
    Uses sign (+/-) where it makes sense. impact can be None.
    """
    f = feat.lower()
    pos = isinstance(impact, (float, int)) and impact > 0
    neg = isinstance(impact, (float, int)) and impact < 0

    # Amount-like
    if "amount" in f:
        if impact is None:
            return "transaction amount deviating from the customer's usual spend"
        if pos:
            return "a higher-than-usual transaction amount versus the customer's history"
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
        return "a change in city or location compared with the customer's typical profile"
    if "country" in f:
        return "activity originating from an unexpected country for this customer"

    # Merchant / MCC
    if "mcc" in f or "merchant" in f:
        return "spend at a merchant or category that is not typical for this customer"

    # Fallback
    return f"behaviour linked to '{feat.replace('_', ' ')}' that deviates from normal patterns"


def shap_to_reasons(pairs: List[Tuple[str, float]], max_reasons: int = 3) -> List[str]:
    """
    Convert list of (feat, impact) into up to max_reasons short reason-phrases.

    Logic:
      - Prefer features with positive SHAP (risk-raising).
      - If none are positive, use largest |impact|.
      - If no numeric impact, use just the feature names.
      - Deduplicate identical reason strings.
    """
    if not isinstance(pairs, list) or len(pairs) == 0:
        return []

    numeric_pairs = [(f, v) for f, v in pairs if isinstance(v, (float, int))]
    none_pairs = [(f, v) for f, v in pairs if not isinstance(v, (float, int))]

    # Prefer positive contributions
    pos_pairs = [(f, v) for f, v in numeric_pairs if v > 0]
    if pos_pairs:
        candidates = sorted(pos_pairs, key=lambda x: -abs(x[1]))
    elif numeric_pairs:
        # fallback: largest |impact|
        candidates = sorted(numeric_pairs, key=lambda x: -abs(x[1]))
    else:
        # only None-valued impacts
        candidates = none_pairs

    chosen = candidates[:max_reasons]

    # Map to reason phrases
    raw_reasons = [describe_feature_with_sign(f, v) for f, v in chosen]

    # Deduplicate while preserving order
    seen = set()
    reasons = []
    for r in raw_reasons:
        if r not in seen:
            reasons.append(r)
            seen.add(r)

    return reasons


# -------------------------------------------------------------------
# 3) 20 templates for silver labels (target_text)
#    - This is exactly your original style, but cleaned & robust
# -------------------------------------------------------------------

def build_explanation_from_reasons(reasons: List[str]) -> str:
    """
    Use up to 20 templates to generate a natural explanation
    from a list of short reason phrases.

    This is used as the "silver label" target_text for SLM training.
    """
    if not reasons:
        return "No material anomaly indicators were detected for this transaction."

    r1 = reasons[0]
    r2 = reasons[1] if len(reasons) > 1 else None
    r3 = reasons[2] if len(reasons) > 2 else None

    joined_all = ", ".join(reasons)
    joined_two = ", and ".join(reasons[:2]) if len(reasons) >= 2 else r1

    templates = [
        lambda: f"The transaction appears anomalous primarily due to {r1}.",
        lambda: f"The alert is driven by {joined_two}, which deviates from the customer's normal behaviour.",
        lambda: f"Risk is elevated because the system detected {joined_all}.",
        lambda: f"This transaction stands out versus historical patterns, especially due to {r1}.",
        lambda: f"Anomaly drivers include {joined_all}, indicating behaviour outside expected norms.",
        lambda: f"The case was flagged as the model identified {joined_two} as unusual for this customer.",
        lambda: f"Model explanations highlight {joined_all} as key signals of potential fraud.",
        lambda: f"The transaction triggers concern due to {r1}, supported by additional irregularities in recent activity.",
        lambda: f"From an analytical standpoint, {joined_two} are the main factors contributing to the alert.",
        lambda: f"The model assigns higher risk because it observes {joined_all} compared with prior behaviour.",
        lambda: f"Behaviour on this transaction diverges from the customer's profile, notably through {r1}.",
        lambda: f"The anomaly score is influenced by {joined_two}, which are not consistent with historical trends.",
        lambda: f"Several patterns, including {joined_all}, differentiate this transaction from typical activity.",
        lambda: f"Alert generation is largely explained by {r1}, with additional contribution from other secondary signals.",
        lambda: f"The system highlights {joined_two} as the strongest indicators of risk relative to this customer's baseline.",
        lambda: f"Compared to usual behaviour, this transaction shows {joined_all}, justifying closer review.",
        lambda: f"The model explanation points to {r1} as the primary driver, with further deviation in related attributes.",
        lambda: f"Key features shaping the fraud decision include {joined_all}, all of which depart from normal patterns.",
        lambda: f"The transaction is flagged because {r1} represents an atypical pattern for this customer.",
        lambda: f"Overall, {joined_all} together create a risk pattern that warranted an anomaly alert.",
    ]

    template_fn = random.choice(templates)
    return template_fn()


# -------------------------------------------------------------------
# 4) Prompt builder (what SLM sees as input_text)
# -------------------------------------------------------------------

def _format_shap_pairs_for_prompt(pairs: List[Tuple[str, float]], max_pairs: int = 5) -> str:
    if not isinstance(pairs, list) or len(pairs) == 0:
        return "not available"

    parts = []
    for feat, impact in pairs[:max_pairs]:
        nice = feat.replace("_", " ")
        if isinstance(impact, (float, int)):
            parts.append(f"{nice} ({impact:+.3f})")
        else:
            parts.append(nice)
    return ", ".join(parts)


def build_generative_prompt_from_row(row: pd.Series) -> str:
    """
    Build the full generative prompt for the SLM.

    Uses:
      - transaction details
      - SHAP drivers (with sign)
      - high-level natural-language cues (_shap_reasons)
    """
    pairs = row.get("_top_shap_pairs") or []
    reasons = row.get("_shap_reasons") or []
    Risk score: {row.get('risk_score')} or []
    Recommended action: {row.get('recommended_action')} or []

    shap_text = _format_shap_pairs_for_prompt(pairs, max_pairs=5)
    reasons_text = "; ".join(reasons[:3]) if reasons else "not available"

    prompt = (
        "You are a fraud investigator assistant. Your task is to write a short, natural-sounding case note "
        "(1–2 sentences) explaining why the transaction below might look anomalous.\n\n"
        "Transaction details:\n"
        f"- Amount: ${row.get('txn_amount')}\n"
        f"- Location: {row.get('txn_city')}, {row.get('txn_country')}\n"
        f"- Minutes since previous transaction: {row.get('mins_since_prev_txn')}\n"
        f"- Transactions in last 24 hours: {row.get('txn_cnt_inlast1440mins')}\n\n"
        "Model explanation signals (from SHAP):\n"
        f"{shap_text}\n\n"
        "High-level anomaly cues you may rely on:\n"
        f"{reasons_text}\n\n"
        "Write your note in your own words. Do not list feature names or SHAP values; instead, describe the behaviour "
        "in plain language as you would to another fraud analyst."
    )
    return prompt


# -------------------------------------------------------------------
# 5) Build SLM training dataset from SHAP-enriched CSV
# -------------------------------------------------------------------

def build_slm_dataset() -> pd.DataFrame:
    """
    Load SHAP-enriched dataframe from processed_shap_path(), build:
      - _top_shap_pairs  ([(feature, impact), ...])
      - _shap_reasons    ([string, string, ...])
      - target_text      (20-template silver label)
      - input_text       (prompt to SLM)

    Save to slm_training_data_path() for training.
    """
    # Make randomness reproducible
    set_seed()

    df = pd.read_csv(processed_shap_path())
    print(f"Loaded SHAP-enriched data from {processed_shap_path()}, shape {df.shape}")

    # 1) Parse SHAP → pairs
    df["_top_shap_pairs"] = df.apply(parse_top_shap, axis=1)

    # 2) Pairs → list of short reasons
    df["_shap_reasons"] = df["_top_shap_pairs"].apply(shap_to_reasons)

    # 3) Reasons → one silver label explanation (target_text)
    df["target_text"] = df["_shap_reasons"].apply(build_explanation_from_reasons)

    # 4) Build prompts
    df["input_text"] = df.apply(build_generative_prompt_from_row, axis=1)

    # 5) Save SLM training dataset
    ensure_dir(DATA_PROCESSED_DIR)
    out_path = slm_training_data_path()
    df.to_csv(out_path, index=False)
    print(f"SLM training dataset saved to {out_path}, shape {df.shape}")

    return df


if __name__ == "__main__":
    build_slm_dataset()
