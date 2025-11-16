# AI-Driven-Fraud-Case-Summarizer
A pipeline that converts structured fraud detection outputs into human-readable case summaries using a small Transformer (Flan-T5) enriched with XGBoost+SHAP interpretability.

# Fraud Case Summarizer — SLM + XGBoost + SHAP
Short: A pipeline that converts structured fraud detection outputs into human-readable case summaries using a small Transformer (Flan-T5) enriched with XGBoost+SHAP interpretability.

# Table of Contents
1. Introduction & Objective
2. Environment Setup
   - Python version & key libraries
   - GPU requirements
3. Data Ingestion
   - Source (Kaggle dataset)
   - Quick EDA & schema
4. Preprocessing & Cleaning
   - Amount normalization
   - Date/time conversion
   - Country/state mapping
5. Feature Engineering (Structured)
   - Time features (hour sin/cos)
   - Amount ratios and delta
   - Velocity & recency features
   - Categorical encodings
6. Customer-level Feature Aggregation
   - Per-customer aggregates & baselines
   - Derived business metrics
7. Interpretability Layer: XGBoost + SHAP
   - Binary anomaly labeling (heuristic)
   - XGBoost training & evaluation
   - SHAP global & per-row explanations
8. SLM Input Construction
   - Base contextual prompt
   - Structured features & SHAP enrichment
9. SLM Training (Flan-T5)
   - Tokenization
   - Seq2SeqTrainer config
   - Training & checkpoints
10. Generation & Post-processing
    - Batch generation
    - Storing outputs
11. Evaluation & Examples
    - Manual examples & ROUGE/qualitative checks
    - Alignment checks (does SLM reflect SHAP? consistency)
12. Deployment / Integration Notes
    - Inference script
    - How to integrate into case management
13. Future Work
    - Better label curation
    - Calibration with real analyst feedback
    - Fine-grained city-country geolocation
14. Files
    - `transactions_with_summaries.csv`
    - `xgb_anomaly_model.json`
    - `shap_global_top20.csv`
15. Author & License
