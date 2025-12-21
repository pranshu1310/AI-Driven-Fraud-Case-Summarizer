#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .shap_enrichment import (
    parse_top_shap,
    shap_to_reasons,
    build_explanation_from_reasons,
    build_generative_prompt_from_row,
)
from .utils import OUTPUTS_DIR, ensure_dir, processed_shap_path, MODELS_DIR


def load_slm_model(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    return tokenizer, model, device


def generate_batch(
    tokenizer,
    model,
    device,
    prompts: List[str],
    max_length: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 8,
) -> List[str]:
    model.eval()
    outputs = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.05,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def run_inference_on_df(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """
    Given a shap-enriched dataframe (e.g., from data/processed/transactions_with_shap.csv)
    that has at least:
      - txn_amount, txn_city, txn_country
      - mins_since_prev_txn, txn_cnt_inlast1440mins
      - shap_summary and/or top_shap_features

    This function:
      - recomputes _top_shap_pairs
      - builds _shap_reasons
      - builds target_text (rule-based silver explanation)
      - builds input_text (prompt)
      - runs the SLM
      - writes generated_narrative
      - saves everything to outputs/summarized_cases.csv
    """
    tokenizer, model, device = load_slm_model(model_dir)

    df = df.copy()

    # if risk_score exists, keep; otherwise compute a fallback from probability-like columns if present
    if 'risk_score' not in df.columns and 'risk_score_prob' in df.columns:
        df['risk_score'] = (df['risk_score_prob'] * 100).round(2)
    # fill missing recommended_action with mapping
    if 'recommended_action' not in df.columns:
        df['recommended_action'] = df['risk_score'].apply(lambda s: "MONITOR / NO ACTION" if s < 40 else ("REVIEW / MONITOR" if s < 60 else ("HOLD & MANUAL REVIEW" if s < 80 else "BLOCK & INVESTIGATE")))


    # 1) Parse SHAP → pairs
    df["_top_shap_pairs"] = df.apply(parse_top_shap, axis=1)

    # 2) Pairs → reasons
    df["_shap_reasons"] = df["_top_shap_pairs"].apply(shap_to_reasons)

    # 3) Reasons → rule-based reference explanation (target_text)
    df["target_text"] = df["_shap_reasons"].apply(build_explanation_from_reasons)

    # 4) Build prompts
    df["input_text"] = df.apply(build_generative_prompt_from_row, axis=1)

    # 5) Generate narratives from SLM
    prompts = df["input_text"].tolist()
    generations = generate_batch(tokenizer, model, device, prompts)
    df["generated_narrative"] = generations

    # 6) Save
    ensure_dir(OUTPUTS_DIR)
    out_path = f"{OUTPUTS_DIR}/summarized_cases.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved generated narratives to {out_path}, shape {df.shape}")

    return df


if __name__ == "__main__":
    # Default: run on full SHAP-enriched dataset
    df_shap = pd.read_csv(processed_shap_path(),low_memory=False)
    df_shap = df_shap.loc[:, ~df_shap.columns.str.startswith("Unnamed")]
    model_dir = f"{MODELS_DIR}/slm_model"  # same as in training
    run_inference_on_df(df_shap, model_dir)
