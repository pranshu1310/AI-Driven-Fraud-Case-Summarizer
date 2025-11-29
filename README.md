# AI-Driven-Fraud-Case-Summarizer
A pipeline that converts structured fraud detection outputs into human-readable case summaries using a small Transformer (Flan-T5) enriched with XGBoost+SHAP interpretability.

# Generative AI – SLM-based Fraud Case Narrative Generator

This project builds an end-to-end pipeline that:

1. Ingests card transactions data and applies feature engineering.
2. Trains an XGBoost anomaly classifier and explains predictions with SHAP.
3. Converts SHAP outputs into natural-language "silver labels".
4. Fine-tunes a small language model (Flan-T5) to generate fraud case narratives for each transaction.

### Key folders

- `data/raw/` – raw CSV (e.g., `transactions_raw.csv`).
- `data/processed/` – feature-engineered and SHAP-enriched datasets.
- `models/` – saved XGBoost model, SHAP global importance, and SLM model.
- `outputs/` – generated narratives for manual review.

### Main scripts (under `src/`)

- `data_preprocessing.py` – load and clean raw data.
- `feature_engineering.py` – build structured features for modeling.
- `anomaly_labeling.py` – create binary anomaly labels (or reuse existing).
- `xgb_interpret.py` – train XGBoost, compute SHAP global and local explanations.
- `shap_enrichment.py` – convert SHAP into prompts and target texts for SLM.
- `slm_train.py` – fine-tune a sequence-to-sequence model (Flan-T5).
- `inference.py` – generate investigation-ready fraud narratives on new data.

You can reproduce the full pipeline by running these modules in order:

```bash
python -m src.data_preprocessing
python -m src.feature_engineering
python -m src.anomaly_labeling
python -m src.xgb_interpret
python -m src.shap_enrichment
python -m src.slm_train
python -m src.inference

