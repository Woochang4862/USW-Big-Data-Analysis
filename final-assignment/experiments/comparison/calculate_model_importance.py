#!/usr/bin/env python
"""
기존 앙상블 결과를 기반으로 각 모델의 중요도 계산
각 모델의 가중치와 개별 성능을 기반으로 제거 시 예상 성능 변화 계산
"""

import json
from pathlib import Path

# 기준 성능 (5개 모델 모두)
baseline_path = Path("results/ensemble_results_with_catboost.json")
with open(baseline_path, "r") as f:
    baseline = json.load(f)

baseline_score = baseline["ensemble_result"]["score"]
baseline_f1 = baseline["ensemble_result"]["f1"]
baseline_auroc = baseline["ensemble_result"]["auroc"]
baseline_weights = baseline["weights"]
individual_results = baseline["individual_results"]

print("=" * 70)
print("모델 중요도 분석 (가중치 기반)")
print("=" * 70)
print(f"\n기준 성능 (5개 모델 모두):")
print(f"  Score: {baseline_score:.6f}")
print(f"  F1: {baseline_f1:.6f}")
print(f"  AUROC: {baseline_auroc:.6f}")

print("\n개별 모델 성능 및 가중치:")
print("-" * 70)
print(f"{'모델':<20} {'개별 Score':<12} {'가중치':<12} {'기여도'}")
print("-" * 70)

model_contributions = {}
for model, weight in baseline_weights.items():
    individual_score = individual_results[model]["score"]
    contribution = weight * individual_score
    model_contributions[model] = {
        "weight": weight,
        "individual_score": individual_score,
        "contribution": contribution,
    }
    print(f"{model:<20} {individual_score:<12.6f} {weight:<12.4f} {contribution:.6f}")

# 가중치가 가장 낮은 모델 = 중요도가 가장 낮은 모델
least_important = min(baseline_weights.items(), key=lambda x: x[1])
most_important = max(baseline_weights.items(), key=lambda x: x[1])

print("\n" + "=" * 70)
print("결론 (가중치 기반)")
print("=" * 70)
print(f"✓ 가장 중요도가 낮은 모델: {least_important[0]}")
print(f"  가중치: {least_important[1]:.4f}")
print(f"  개별 Score: {individual_results[least_important[0]]['score']:.6f}")
print(f"\n✓ 가장 중요도가 높은 모델: {most_important[0]}")
print(f"  가중치: {most_important[1]:.4f}")
print(f"  개별 Score: {individual_results[most_important[0]]['score']:.6f}")

# 실제 제거 실험 결과가 있다면 확인
print("\n" + "=" * 70)
print("실제 제거 실험 결과 확인")
print("=" * 70)

models = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
case_names = {
    "LogisticRegression": "remove_logistic",
    "RandomForest": "remove_randomforest",
    "XGBoost": "remove_xgboost",
    "LightGBM": "remove_lightgbm",
    "CatBoost": "remove_catboost",
}

removal_results = {}
for model in models:
    case = case_names[model]
    result_path = Path(f"results/ensemble_results_{case}.json")
    if result_path.exists():
        with open(result_path, "r") as f:
            result = json.load(f)
            removal_results[model] = {
                "score": result["ensemble_result"]["score"],
                "f1": result["ensemble_result"]["f1"],
                "auroc": result["ensemble_result"]["auroc"],
            }

if removal_results:
    print("\n각 모델 제거 시 실제 성능 변화:")
    print("-" * 70)
    print(f"{'제거된 모델':<20} {'Score':<12} {'Score 변화':<15} {'F1 변화':<12} {'AUROC 변화'}")
    print("-" * 70)

    score_changes = {}
    for model, result in removal_results.items():
        score_diff = result["score"] - baseline_score
        f1_diff = result["f1"] - baseline_f1
        auroc_diff = result["auroc"] - baseline_auroc
        score_changes[model] = score_diff

        print(
            f"{model:<20} {result['score']:<12.6f} "
            f"{score_diff:+.6f}        {f1_diff:+.6f}    {auroc_diff:+.6f}"
        )

    if score_changes:
        least_important_actual = max(score_changes.items(), key=lambda x: x[1])
        most_important_actual = min(score_changes.items(), key=lambda x: x[1])

        print("\n" + "=" * 70)
        print("실제 실험 결과 결론")
        print("=" * 70)
        print(f"✓ 가장 중요도가 낮은 모델: {least_important_actual[0]}")
        print(f"  제거 시 Score 변화: {least_important_actual[1]:+.6f}")
        print(f"  제거 후 Score: {removal_results[least_important_actual[0]]['score']:.6f}")
        print(f"\n✓ 가장 중요도가 높은 모델: {most_important_actual[0]}")
        print(f"  제거 시 Score 변화: {most_important_actual[1]:+.6f}")
        print(f"  제거 후 Score: {removal_results[most_important_actual[0]]['score']:.6f}")
else:
    print("\n실제 제거 실험 결과가 아직 없습니다.")
    print("각 모델을 제거한 실험을 실행하려면:")
    print("  python experiments/ensemble_experiment.py --case remove_logistic --exclude LogisticRegression")
    print("  python experiments/ensemble_experiment.py --case remove_randomforest --exclude RandomForest")
    print("  python experiments/ensemble_experiment.py --case remove_xgboost --exclude XGBoost")
    print("  python experiments/ensemble_experiment.py --case remove_lightgbm --exclude LightGBM")
    print("  python experiments/ensemble_experiment.py --case remove_catboost --exclude CatBoost")

print("=" * 70)


