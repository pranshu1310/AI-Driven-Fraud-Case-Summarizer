#!/usr/bin/env python
# coding: utf-8

from typing import List

import pandas as pd
import torch
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
        batch = prompts[i : i + batch_size]
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
                early_stopping=True,
                repetition_penalty=1.05,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def run_inference_on_df(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """
    Given a SHAP-enriched dataframe that has at least:
      - txn_amount, txn_city, txn_country
      - mins_since_prev_txn, txn_cnt_inlast1440mins
      - top_shap_features, shap_summary

    this will:
      - reconstruct _top_shap_pairs
      - build _shap_reasons
      - build input_text (prompt)
      - build target_text (silver explanation)
      - generate 'generated_narrative' via SLM
      - save everything to outputs/summarized_cases.csv
    """
    tokenizer, model, device = load_slm_model(model_dir)

    df = df.copy()

    # 1) Parse SHAP
    df["_top_shap_pairs"] = df.apply(parse_top_shap, axis=1)

    # 2) SHAP -> reasons
    df["_shap_reasons"] = df["_top_shap_pairs"].apply(shap_to_reasons)

    # 3) Silver explanation (same as training target_text logic)
    df["target_text"] = df["_shap_reasons"].apply(build_explanation_from_reasons)

    # 4) Build prompts
    df["input_text"] = df.apply(build_generative_prompt_from_row, axis=1)

    prompts = df["input_text"].tolist()
    generations = generate_batch(tokenizer, model, device, prompts)

    df["generated_narrative"] = generations

    # Save final output
    ensure_dir(OUTPUTS_DIR)
    out_path = f"{OUTPUTS_DIR}/summarized_cases.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved generated narratives to {out_path}, shape {df.shape}")

    return df


if __name__ == "__main__":
    # Convenience: run inference on the SHAP-enriched dataset
    shap_path = processed_shap_path()
    df_shap = pd.read_csv(shap_path)
    model_dir = f"{MODELS_DIR}/slm_model"
    run_inference_on_df(df_shap, model_dir)
