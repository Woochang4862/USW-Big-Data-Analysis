#!/usr/bin/env python
"""
5개 모델 앙상블에서 각 모델의 중요도 측정
각 모델을 하나씩 제거하고 나머지 4개 모델로 앙상블 구성하여 성능 비교
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_feature_label_pairs, ensure_data_dirs
from src.data.preprocessing import infer_categorical_from_dtype, impute_by_rules
from src.ensemble import weighted_ensemble, find_best_models
from src.models.factory import create_model
from src.utils.metrics import evaluate_model

# 5개 모델 리스트
all_models = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]

def run_ensemble_experiment(exclude_models=None):
    """앙상블 실험 실행"""
    base_dir = project_root
    raw_dir, _ = ensure_data_dirs(base_dir)
    results_path = base_dir / "results" / "overfitting_experiments.json"
    
    # 최고 모델 찾기
    best_models = find_best_models(results_path, exclude_models=exclude_models)
    
    # 데이터 로드 및 전처리
    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    categorical_cols = infer_categorical_from_dtype(X_train)
    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)
    
    # 모델 학습
    trained_models = {}
    for model_type, info in best_models.items():
        params = info["params"].copy()
        params["random_state"] = 42
        model = create_model(model_type, params)
        model.fit(X_train_imp, y_train)
        trained_models[model_type] = model
    
    # 앙상블 예측
    ensemble_prob, normalized_weights = weighted_ensemble(
        best_models, X_train_imp, y_train, X_test_imp
    )
    
    # 평가
    ensemble_metrics = evaluate_model(y_test, ensemble_prob)
    ensemble_score = (ensemble_metrics["f1"] + ensemble_metrics["auroc"]) / 2
    
    return {
        "score": ensemble_score,
        "f1": ensemble_metrics["f1"],
        "auroc": ensemble_metrics["auroc"],
        "models": list(best_models.keys()),
        "weights": normalized_weights,
    }

# 기준 성능 (5개 모델 모두)
print("=" * 70)
print("기준 성능: 5개 모델 앙상블")
print("=" * 70)
baseline_result = run_ensemble_experiment(exclude_models=None)
baseline_score = baseline_result["score"]

print(f"모델: {baseline_result['models']}")
print(f"Score: {baseline_score:.6f}")
print(f"F1: {baseline_result['f1']:.6f}")
print(f"AUROC: {baseline_result['auroc']:.6f}")

# 각 모델을 하나씩 제거하며 실험
results = {}

for model_to_remove in all_models:
    print("\n" + "=" * 70)
    print(f"{model_to_remove} 제거 실험")
    print("=" * 70)
    
    exclude_models = [model_to_remove]
    result = run_ensemble_experiment(exclude_models=exclude_models)
    results[model_to_remove] = result
    
    print(f"모델: {result['models']}")
    print(f"Score: {result['score']:.6f} (변화: {result['score'] - baseline_score:+.6f})")
    print(f"F1: {result['f1']:.6f} (변화: {result['f1'] - baseline_result['f1']:+.6f})")
    print(f"AUROC: {result['auroc']:.6f} (변화: {result['auroc'] - baseline_result['auroc']:+.6f})")

# 결과 비교
print("\n" + "=" * 70)
print("모델 중요도 분석 결과")
print("=" * 70)
print(f"\n기준 성능 (5개 모델 모두):")
print(f"  Score: {baseline_score:.6f}")
print(f"  F1: {baseline_result['f1']:.6f}")
print(f"  AUROC: {baseline_result['auroc']:.6f}")

print("\n각 모델 제거 시 성능 변화:")
print("-" * 70)
print(f"{'제거된 모델':<20} {'Score':<12} {'Score 변화':<15} {'F1 변화':<12} {'AUROC 변화'}")
print("-" * 70)

score_changes = {}
for model, result in results.items():
    score_diff = result["score"] - baseline_score
    f1_diff = result["f1"] - baseline_result["f1"]
    auroc_diff = result["auroc"] - baseline_result["auroc"]
    score_changes[model] = score_diff
    
    print(
        f"{model:<20} {result['score']:<12.6f} "
        f"{score_diff:+.6f}        {f1_diff:+.6f}    {auroc_diff:+.6f}"
    )

# 가장 중요도가 낮은 모델 (제거해도 성능 저하가 가장 적은 모델)
least_important = max(score_changes.items(), key=lambda x: x[1])
most_important = min(score_changes.items(), key=lambda x: x[1])

print("\n" + "=" * 70)
print("결론")
print("=" * 70)
print(f"✓ 가장 중요도가 낮은 모델: {least_important[0]}")
print(f"  제거 시 Score 변화: {least_important[1]:+.6f}")
print(f"  제거 후 Score: {results[least_important[0]]['score']:.6f}")
print(f"\n✓ 가장 중요도가 높은 모델: {most_important[0]}")
print(f"  제거 시 Score 변화: {most_important[1]:+.6f}")
print(f"  제거 후 Score: {results[most_important[0]]['score']:.6f}")

# 결과 저장
comparison_result = {
    "baseline": {
        "models": baseline_result["models"],
        "score": baseline_score,
        "f1": baseline_result["f1"],
        "auroc": baseline_result["auroc"],
    },
    "removed_models": {
        model: {
            "remaining_models": result["models"],
            "score": result["score"],
            "f1": result["f1"],
            "auroc": result["auroc"],
            "score_change": result["score"] - baseline_score,
            "f1_change": result["f1"] - baseline_result["f1"],
            "auroc_change": result["auroc"] - baseline_result["auroc"],
        }
        for model, result in results.items()
    },
    "least_important_model": least_important[0],
    "most_important_model": most_important[0],
}

output_path = project_root / "results" / "model_importance_analysis.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(comparison_result, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장: {output_path}")
print("=" * 70)


