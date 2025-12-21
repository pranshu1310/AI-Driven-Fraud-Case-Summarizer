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
# -------------------------------------------------------------------

def parse_top_shap(row) -> List[Tuple[str, float]]:
    """
    Returns list of (feature, impact) pairs.

    Priority:
      1) shap_summary with numeric sign
      2) top_shap_features list or stringified list
      3) empty list
    """
    # ---- 1) Prefer shap_summary (numeric, signed) ----
    summary = row.get("shap_summary", "")
    if isinstance(summary, str) and "Top factors" in summary:
        parts = re.findall(r"([A-Za-z0-9_]+)\s*([+-]?\d+\.\d+)", summary)
        if parts:
            return [(feat, float(val)) for feat, val in parts]

    # ---- 2) Fallback: top_shap_features ----
    raw = row.get("top_shap_features")

    if isinstance(raw, list):
        return [(str(x), None) for x in raw]

    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [(str(x), None) for x in parsed]
        except Exception:
            pass

    return []


# -------------------------------------------------------------------
# 2) Feature → short natural-language description (sign-aware)
# -------------------------------------------------------------------

def describe_feature_with_sign(feat: str, impact):
    f = feat.lower()
    pos = isinstance(impact, (float, int)) and impact > 0
    neg = isinstance(impact, (float, int)) and impact < 0

    if "amount" in f:
        if pos:
            return "a higher-than-usual transaction amount compared with this customer's history"
        if neg:
            return "a lower-than-usual transaction amount compared with typical behaviour"
        return "an unusual transaction amount relative to the customer's normal spend"

    if "cnt" in f or "freq" in f:
        if pos:
            return "a spike in recent transaction frequency"
        if neg:
            return "a reduction in transaction frequency compared to normal patterns"
        return "an unusual pattern in transaction frequency"

    if "mins_since_prev_txn" in f or "mins" in f:
        if pos:
            return "a very short gap since the previous transaction"
        if neg:
            return "an unusually long gap since the previous transaction"
        return "irregular spacing between consecutive transactions"

    if "hour" in f or "time" in f:
        return "the transaction occurring at an unusual time of day for this customer"

    if "city" in f:
        return "activity occurring in a city that differs from the customer's usual locations"

    if "country" in f:
        return "activity originating from an unexpected country for this customer"

    if "mcc" in f or "merchant" in f:
        return "spend at a merchant category that is not typical for this customer"

    return f"behaviour related to '{feat.replace('_', ' ')}' that deviates from normal patterns"


def shap_to_reasons(
    pairs: List[Tuple[str, float]],
    max_reasons: int = 3,
) -> List[str]:
    """
    Convert SHAP (feature, impact) pairs into short, human-readable reasons.
    """
    if not pairs:
        return []

    numeric = [(f, v) for f, v in pairs if isinstance(v, (float, int))]
    non_numeric = [(f, v) for f, v in pairs if not isinstance(v, (float, int))]

    # Prefer positive (risk-increasing) contributions
    positives = [(f, v) for f, v in numeric if v > 0]

    if positives:
        chosen = sorted(positives, key=lambda x: -abs(x[1]))
    elif numeric:
        chosen = sorted(numeric, key=lambda x: -abs(x[1]))
    else:
        chosen = non_numeric

    chosen = chosen[:max_reasons]

    reasons = []
    seen = set()
    for f, v in chosen:
        r = describe_feature_with_sign(f, v)
        if r not in seen:
            reasons.append(r)
            seen.add(r)

    return reasons


# -------------------------------------------------------------------
# 3) Silver-label explanation (20 templates)
# -------------------------------------------------------------------

def build_explanation_from_reasons(reasons: List[str]) -> str:
    if not reasons:
        return "No material anomaly indicators were detected for this transaction."

    r1 = reasons[0]
    r2 = reasons[1] if len(reasons) > 1 else None
    r3 = reasons[2] if len(reasons) > 2 else None

    joined_all = ", ".join(reasons)
    joined_two = ", and ".join(reasons[:2])

    templates = [
        lambda: f"The transaction appears anomalous primarily due to {r1}.",
        lambda: f"The alert is driven by {joined_two}, which deviates from the customer's normal behaviour.",
        lambda: f"Risk is elevated because the system detected {joined_all}.",
        lambda: f"This transaction stands out from historical patterns, especially due to {r1}.",
        lambda: f"Anomaly drivers include {joined_all}, indicating behaviour outside expected norms.",
        lambda: f"The case was flagged as the model identified {joined_two} as unusual for this customer.",
        lambda: f"Model explanations highlight {joined_all} as key signals of potential fraud.",
        lambda: f"The transaction triggers concern due to {r1}, supported by additional irregularities.",
        lambda: f"From an analytical standpoint, {joined_two} are the main contributors to the alert.",
        lambda: f"The model assigns higher risk because it observes {joined_all} compared with prior behaviour.",
        lambda: f"Behaviour diverges from the customer's profile, notably through {r1}.",
        lambda: f"The anomaly score is influenced by {joined_two}, inconsistent with historical trends.",
        lambda: f"Several patterns, including {joined_all}, differentiate this transaction from typical activity.",
        lambda: f"Alert generation is largely explained by {r1}, with secondary signals reinforcing the risk.",
        lambda: f"The system highlights {joined_two} as the strongest indicators of elevated risk.",
        lambda: f"Compared to usual behaviour, this transaction shows {joined_all}, justifying review.",
        lambda: f"The model explanation points to {r1} as the primary driver of this alert.",
        lambda: f"Key features shaping the decision include {joined_all}, all departing from normal patterns.",
        lambda: f"The transaction is flagged because {r1} represents atypical customer behaviour.",
        lambda: f"Overall, {joined_all} together form a risk pattern that warranted the alert.",
    ]

    return random.choice(templates)()


# -------------------------------------------------------------------
# 4) Prompt builder (FIXED — includes risk & action)
# -------------------------------------------------------------------

def _format_shap_pairs_for_prompt(
    pairs: List[Tuple[str, float]],
    max_pairs: int = 5,
) -> str:
    if not pairs:
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
    pairs = row.get("_top_shap_pairs") or []
    reasons = row.get("_shap_reasons") or []

    risk_score = row.get("risk_score", "NA")
    recommended_action = row.get("recommended_action", "NA")

    shap_text = _format_shap_pairs_for_prompt(pairs)
    reasons_text = "; ".join(reasons) if reasons else "not available"

    prompt = (
        "You are a fraud investigator assistant. Your task is to write a short, natural-sounding case note "
        "(1–2 sentences) explaining why the transaction below might appear anomalous.\n\n"
        "Transaction details:\n"
        f"- Amount: ${row.get('txn_amount')}\n"
        f"- Location: {row.get('txn_city')}, {row.get('txn_country')}\n"
        f"- Minutes since previous transaction: {row.get('mins_since_prev_txn')}\n"
        f"- Transactions in last 24 hours: {row.get('txn_cnt_inlast1440mins')}\n"
        f"- Risk score: {risk_score}\n"
        f"- Recommended action: {recommended_action}\n\n"
        "Model explanation signals (from SHAP):\n"
        f"{shap_text}\n\n"
        "High-level anomaly cues:\n"
        f"{reasons_text}\n\n"
        "Write the note in plain language as you would to another fraud analyst. "
        "Do not mention feature names or SHAP values explicitly."
    )
    return prompt


# -------------------------------------------------------------------
# 5) Build SLM training dataset
# -------------------------------------------------------------------

def build_slm_dataset() -> pd.DataFrame:
    set_seed()

    df = pd.read_csv(processed_shap_path(),low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"Loaded SHAP-enriched data from {processed_shap_path()}, shape={df.shape}")

    df["_top_shap_pairs"] = df.apply(parse_top_shap, axis=1)
    df["_shap_reasons"] = df["_top_shap_pairs"].apply(shap_to_reasons)
    df["target_text"] = df["_shap_reasons"].apply(build_explanation_from_reasons)
    df["input_text"] = df.apply(build_generative_prompt_from_row, axis=1)

    ensure_dir(DATA_PROCESSED_DIR)
    out_path = slm_training_data_path()
    df.to_csv(out_path, index=False)

    print(f"SLM training dataset saved to {out_path}, shape={df.shape}")
    return df


if __name__ == "__main__":
    build_slm_dataset()
