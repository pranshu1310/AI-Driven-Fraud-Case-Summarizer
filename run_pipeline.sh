#!/bin/bash
set -e

echo "🔵 Step 1: Preprocessing"
python -m src.data_preprocessing

echo "🟢 Step 2: Feature Engineering (transaction-level)"
python -m src.feature_engineering

echo "🟡 Step 3: Customer Profile Features (offline)"
python -m src.customer_profile_features

echo "🟠 Step 4: Anomaly Labeling"
python -m src.anomaly_labeling

echo "🟣 Step 5: XGBoost + SHAP"
python -m src.xgb_interpret

echo "🟤 Step 6: SHAP Enrichment"
python -m src.shap_enrichment

echo "🔴 Step 7: SLM Training"
python -m src.slm_train

echo "🟢 Step 8: Inference"
python -m src.inference

echo "✅ Pipeline completed successfully!"
