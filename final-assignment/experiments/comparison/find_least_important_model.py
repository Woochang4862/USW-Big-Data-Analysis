#!/usr/bin/env python
"""
5개 모델 앙상블에서 각 모델의 중요도 측정
- 각 모델을 하나씩 제거하고 나머지 4개 모델로 앙상블 구성
- 성능 변화를 비교하여 중요도가 가장 낮은 모델 찾기
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import sys

# 5개 모델 리스트
all_models = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]

# 기준 성능은 이미 있는 결과 사용 (ensemble_results_with_catboost.json)
print("=" * 70)
print("기준 성능: 5개 모델 앙상블 (기존 결과 사용)")
print("=" * 70)

# 각 모델을 하나씩 제거하며 실험
results = {}

for model_to_remove in all_models:
    print("\n" + "=" * 70)
    print(f"{model_to_remove} 제거 실험")
    print("=" * 70)
    
    exclude_models = [model_to_remove]
    case_name = f"remove_{model_to_remove.lower()}"
    
    # subprocess로 실행
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "ensemble_experiment.py"),
        "--case", case_name,
        "--exclude", model_to_remove
    ]
    print(f"실행 명령: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    if result.returncode != 0:
        print(f"오류: {result.stderr}")
    else:
        print(result.stdout)
    
    # 결과 로드
    results_path = project_root / "results" / f"ensemble_results_{case_name}.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            result = json.load(f)
            results[model_to_remove] = {
                "score": result["ensemble_result"]["score"],
                "f1": result["ensemble_result"]["f1"],
                "auroc": result["ensemble_result"]["auroc"],
                "models": list(result["best_models"].keys()),
            }

# 기준 성능 로드
baseline_path = project_root / "results" / "ensemble_results_baseline_5models.json"
if baseline_path.exists():
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
        baseline_score = baseline["ensemble_result"]["score"]
        baseline_f1 = baseline["ensemble_result"]["f1"]
        baseline_auroc = baseline["ensemble_result"]["auroc"]
else:
    # 기존 5개 모델 결과 사용
    with open(project_root / "results" / "ensemble_results_with_catboost.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)
        baseline_score = baseline["ensemble_result"]["score"]
        baseline_f1 = baseline["ensemble_result"]["f1"]
        baseline_auroc = baseline["ensemble_result"]["auroc"]

# 결과 비교
print("\n" + "=" * 70)
print("모델 중요도 분석 결과")
print("=" * 70)
print(f"\n기준 성능 (5개 모델 모두):")
print(f"  Score: {baseline_score:.6f}")
print(f"  F1: {baseline_f1:.6f}")
print(f"  AUROC: {baseline_auroc:.6f}")

print("\n각 모델 제거 시 성능 변화:")
print("-" * 70)
print(f"{'제거된 모델':<20} {'Score':<12} {'Score 변화':<15} {'F1 변화':<12} {'AUROC 변화'}")
print("-" * 70)

score_changes = {}
for model, result in results.items():
    score_diff = result["score"] - baseline_score
    f1_diff = result["f1"] - baseline_f1
    auroc_diff = result["auroc"] - baseline_auroc
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
        "models": all_models,
        "score": baseline_score,
        "f1": baseline_f1,
        "auroc": baseline_auroc,
    },
    "removed_models": {
        model: {
            "remaining_models": result["models"],
            "score": result["score"],
            "f1": result["f1"],
            "auroc": result["auroc"],
            "score_change": result["score"] - baseline_score,
            "f1_change": result["f1"] - baseline_f1,
            "auroc_change": result["auroc"] - baseline_auroc,
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

