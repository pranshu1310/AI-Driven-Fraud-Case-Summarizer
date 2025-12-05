#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .shap_enrichment import build_generative_prompt_from_row, parse_shap_summary
from .utils import OUTPUTS_DIR, ensure_dir


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
                early_stopping=True,
                repetition_penalty=1.05,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def run_inference_on_df(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """
    Given a dataframe that has at least:
      - txn_amount, txn_city, txn_country
      - mins_since_prev_txn, txn_cnt_inlast1440mins
      - shap_summary
    build prompts and generate narratives.
    """
    tokenizer, model, device = load_slm_model(model_dir)

    # Parse shap_summary -> _top_shap_pairs
    df = df.copy()
    df["_top_shap_pairs"] = df["shap_summary"].apply(parse_shap_summary)

    prompts = []
    for _, row in df.iterrows():
        pairs = row["_top_shap_pairs"]
        prompts.append(build_generative_prompt_from_row(row, pairs))

    generations = generate_batch(tokenizer, model, device, prompts)
    df["generated_narrative"] = generations

    ensure_dir(OUTPUTS_DIR)
    out_path = f"{OUTPUTS_DIR}/summarized_cases.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved generated narratives to {out_path}, shape {df.shape}")

    return df


if __name__ == "__main__":
    # Example usage: load SHAP-enriched dataset and generate narratives
    from .utils import processed_shap_path, MODELS_DIR
    df_shap = pd.read_csv(processed_shap_path())
    model_dir = f"{MODELS_DIR}/slm_model"   # same as you used in training
    run_inference_on_df(df_shap.head(100), model_dir)

