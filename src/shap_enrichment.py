#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pandas as pd
from typing import List, Tuple

from .utils import processed_shap_path, slm_training_data_path, ensure_dir, DATA_PROCESSED_DIR


# 1) Parse shap_summary => list of (feature, impact)
def parse_shap_summary(summary: str) -> List[Tuple[str, float]]:
    """
    Parse a string like:
        "Top factors: mins_since_prev_txn +0.049, city_enc -0.047, ..."
    into:
        [("mins_since_prev_txn", 0.049), ("city_enc", -0.047), ...]
    """
    if not isinstance(summary, str):
        return []
    if "Top factors" not in summary:
        return []
    parts = re.findall(r"([A-Za-z0-9_]+)\s*([+-]?\d+\.\d+)", summary)
    pairs: List[Tuple[str, float]] = [(p[0], float(p[1])) for p in parts]
    return pairs


# 2) Map features -> human reasons (your feature_reason_map)
feature_reason_map = {
    "amount": "unusually high or irregular transaction amount",
    "ratio": "transaction amount deviation from usual spending pattern",
    "cnt": "sudden spike in transaction frequency",
    "freq": "elevated spending frequency",
    "time": "transaction at unusual timing",
    "mins": "irregular timing pattern",
    "city": "unexpected city/location change",
    "country": "unusual country for this customer",
    "mcc": "merchant category atypical for this customer",
    "device": "new or rare device",
    "merchant": "unusual merchant behavior",
}


def make_anomaly_summary_from_pairs(pairs: List[Tuple[str, float]]) -> str:
    """
    Create a human-friendly 'silver label' explanation from top SHAP pairs.
    """
    if not isinstance(pairs, list) or len(pairs) == 0:
        return "Regular transaction."

    explanations = []
    for feat, impact in pairs[:3]:
        matched = False
        for key, reason in feature_reason_map.items():
            if key in feat:
                explanations.append(reason)
                matched = True
                break
        if not matched:
            explanations.append(f"irregularity in {feat.replace('_', ' ')}")

    reason_text = ", and ".join(explanations)
    return f"This transaction appears suspicious due to {reason_text}."


# 3) Lightweight paraphrase templates (to avoid template overfitting)
def paraphrase_variants(plain_text: str):
    templates = [
        lambda t: t,
        lambda t: t.replace("This transaction appears suspicious due to", "Flagged because of"),
        lambda t: t.replace("This transaction appears suspicious due to", "Suspicion arises from"),
        lambda t: t.replace("This transaction appears suspicious due to", "Potential anomaly:"),
    ]
    variants = []
    for fn in templates[:3]:
        variants.append(fn(plain_text))
    return variants


# 4) Generative prompt builder for SLM
def build_generative_prompt_from_row(row: pd.Series, top_pairs: List[Tuple[str, float]]) -> str:
    # Top 5 shap drivers with sign and magnitude
    shap_text = ", ".join(
        [f"{feat.replace('_',' ')} ({impact:+.3f})" for feat, impact in top_pairs[:5]]
    ) if top_pairs else "none"

    prompt = (
        "You are a fraud investigator assistant. Generate a concise 1–2 line explanation "
        "for why the transaction may be anomalous. Use professional language suitable for case notes.\n\n"
        "Transaction Details:\n"
        f"- Amount: ${row.get('txn_amount')}\n"
        f"- Location: {row.get('txn_city')}, {row.get('txn_country')}\n"
        f"- Minutes since previous transaction: {row.get('mins_since_prev_txn')}\n"
        f"- Transactions in last 24 hours: {row.get('txn_cnt_inlast1440mins')}\n\n"
        "Top SHAP Drivers:\n"
        f"{shap_text}\n\n"
        "Instruction: Using the SHAP drivers, describe the observable anomaly in 1–2 sentences. "
        "Do not enumerate features; produce a single natural-language explanation suitable for investigators."
    )
    return prompt


def build_slm_dataset() -> pd.DataFrame:
    """
    Load SHAP-enriched dataframe, build:
      - _top_shap_pairs: list[(feature, impact)]
      - target_text: silver explanation from SHAP
      - input_text: generative prompt
    with paraphrased targets to encourage diverse generative behavior.
    """
    df = pd.read_csv(processed_shap_path())

    # 1) Parse shap_summary -> _top_shap_pairs so we keep numeric +/- signs
    df["_top_shap_pairs"] = df["shap_summary"].apply(parse_shap_summary)

    # 2) Build base silver target
    df["target_text"] = df["_top_shap_pairs"].apply(make_anomaly_summary_from_pairs)

    # 3) Expand with paraphrases (each row -> 1–3 samples)
    rows = []
    for _, r in df.iterrows():
        pairs = r["_top_shap_pairs"]
        base = r["target_text"]
        variants = paraphrase_variants(base)

        for variant in variants:
            rows.append({
                "txn_id": r.get("txn_id"),
                "txn_amount": r.get("txn_amount"),
                "txn_city": r.get("txn_city"),
                "txn_country": r.get("txn_country"),
                "mins_since_prev_txn": r.get("mins_since_prev_txn"),
                "txn_cnt_inlast1440mins": r.get("txn_cnt_inlast1440mins"),
                "_top_shap_pairs": pairs,
                "input_text": None,  # we fill after
                "target_text": variant,
            })

    df_slm = pd.DataFrame(rows)

    # 4) Build prompts
    def _build_prompt(row):
        pairs = row["_top_shap_pairs"]
        return build_generative_prompt_from_row(row, pairs)

    df_slm["input_text"] = df_slm.apply(_build_prompt, axis=1)
    df_slm["target_text"] = df_slm["target_text"].astype(str)

    ensure_dir(DATA_PROCESSED_DIR)
    df_slm.to_csv(slm_training_data_path(), index=False)
    print(f"SLM training dataset saved to {slm_training_data_path()}, shape {df_slm.shape}")

    return df_slm


if __name__ == "__main__":
    build_slm_dataset()

