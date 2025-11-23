#!/bin/bash

cd /Users/jeong-uchang/USW-Big-Data-Analysis/final-assignment
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate team_project

echo "=========================================="
echo "모델 중요도 분석 실험 시작"
echo "=========================================="

# 각 모델을 하나씩 제거하며 실험
models=("LogisticRegression" "RandomForest" "XGBoost" "LightGBM" "CatBoost")

for model in "${models[@]}"; do
    echo ""
    echo "=========================================="
    echo "$model 제거 실험"
    echo "=========================================="
    python experiments/ensemble_experiment.py --case "remove_${model,,}" --exclude "$model"
done

echo ""
echo "=========================================="
echo "모든 실험 완료!"
echo "=========================================="


