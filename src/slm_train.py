#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch

from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from sklearn.model_selection import train_test_split

from .utils import slm_training_data_path, MODELS_DIR, ensure_dir


def prepare_hf_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 80,
):
    """
    Convert a pandas dataframe with columns input_text, target_text
    into a HuggingFace Dataset with tokenized fields.
    """
    inputs = df["input_text"].astype(str).tolist()
    targets = df["target_text"].astype(str).tolist()

    tokenized_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    labels = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in tokenized_targets["input_ids"]
    ]

    dataset = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,
    }
    return HFDataset.from_dict(dataset)


def train_slm_model(
    model_name: str = "google/flan-t5-small",
    output_subdir: str = "slm_model",
    max_input_length: int = 256,
    max_target_length: int = 80,
    num_train_epochs: int = 3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df_slm = pd.read_csv(slm_training_data_path())
    print("Loaded SLM training data, shape:", df_slm.shape)

    train_df, temp_df = train_test_split(df_slm, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    train_hf = prepare_hf_dataset(
        train_df.reset_index(drop=True),
        tokenizer,
        max_input_length,
        max_target_length,
    )
    val_hf = prepare_hf_dataset(
        val_df.reset_index(drop=True),
        tokenizer,
        max_input_length,
        max_target_length,
    )

    batch_size = 8
    args = Seq2SeqTrainingArguments(
        output_dir=f"./{output_subdir}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-4,
        num_train_epochs=num_train_epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=2,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save under models/slm_model
    slm_dir = f"{MODELS_DIR}/{output_subdir}"
    ensure_dir(slm_dir)
    trainer.save_model(slm_dir)
    tokenizer.save_pretrained(slm_dir)

    print(f"Saved SLM model + tokenizer to {slm_dir}")
    return slm_dir, test_df


if __name__ == "__main__":
    train_slm_model()
