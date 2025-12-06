#!/bin/bash
set -e  # stop if any step fails

echo "🔵 Step 1: Preprocessing"
python -m src.data_preprocessing

echo "🟢 Step 2: Feature Engineering"
python -m src.feature_engineering

echo "🟠 Step 3: Anomaly Labeling"
python -m src.anomaly_labeling

echo "🟣 Step 4: XGBoost + SHAP"
python -m src.xgb_interpret

echo "🟡 Step 5: SHAP Enrichment"
python -m src.shap_enrichment

echo "🔴 Step 6: SLM Training"
python -m src.slm_train

echo "🟤 Step 7: Inference"
python -m src.inference

echo "✅ Pipeline completed successfully!"
