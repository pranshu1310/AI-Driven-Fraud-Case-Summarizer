import os
import numpy as np
import pandas as pd
import random

RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Where things live (you can tweak these)
DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
OUTPUTS_DIR = os.path.join("outputs")

def default_raw_path():
    return os.path.join(DATA_RAW_DIR, "transactions_raw.csv")

def processed_fe_path():
    return os.path.join(DATA_PROCESSED_DIR, "transactions_fe.csv")

def processed_shap_path():
    return os.path.join(DATA_PROCESSED_DIR, "transactions_with_shap.csv")

def slm_training_data_path():
    return os.path.join(DATA_PROCESSED_DIR, "slm_training_data.csv")
