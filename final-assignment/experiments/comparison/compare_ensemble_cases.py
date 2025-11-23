#!/usr/bin/env python
"""
앙상블 케이스 비교 실험
- 케이스 1: 3개 모델 앙상블 (LogisticRegression, RandomForest, XGBoost)
- 케이스 2: 2개 모델 앙상블 (3개 중 가장 낮은 스코어 모델 제거)
각 케이스를 실행하고 JSON 결과를 비교
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_ensemble_case(case_name: str, exclude_models: list = None):
    """앙상블 케이스 실행"""
    base_dir = Path(__file__).resolve().parent
    
    cmd = [sys.executable, "ensemble_best_models.py", "--case", case_name]
    if exclude_models:
        cmd.extend(["--exclude"] + exclude_models)
    
    result = subprocess.run(
        cmd,
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"케이스 {case_name} 실행 실패:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def load_ensemble_result(results_dir: Path, case_name: str) -> dict:
    """앙상블 결과 JSON 로드"""
    result_path = results_dir / f"ensemble_results_{case_name}.json"
    if not result_path.exists():
        return None
    
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_lowest_score_model(best_models: dict) -> str:
    """가장 낮은 test_score를 가진 모델 찾기"""
    lowest_score = float('inf')
    lowest_model = None
    
    for model_type, info in best_models.items():
        if info["test_score"] < lowest_score:
            lowest_score = info["test_score"]
            lowest_model = model_type
    
    return lowest_model


def compare_ensemble_cases():
    """앙상블 케이스 비교"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    
    print("=" * 70)
    print("앙상블 케이스 비교 실험")
    print("=" * 70)
    
    # 케이스 1: 3개 모델 앙상블 (DecisionTree 제외)
    print("\n" + "=" * 70)
    print("케이스 1: 3개 모델 앙상블 (LogisticRegression, RandomForest, XGBoost)")
    print("=" * 70)
    
    success1 = run_ensemble_case("case1", exclude_models=["DecisionTree", "NeuralNetwork"])
    if not success1:
        print("케이스 1 실행 실패")
        return
    
    case1_result = load_ensemble_result(results_dir, "case1")
    if case1_result is None:
        print("케이스 1 결과 파일을 찾을 수 없습니다.")
        return
    
    # 케이스 1에서 가장 낮은 스코어 모델 찾기
    lowest_model = find_lowest_score_model(case1_result["best_models"])
    print(f"\n케이스 1에서 가장 낮은 스코어 모델: {lowest_model}")
    print(f"  Score: {case1_result['best_models'][lowest_model]['test_score']:.4f}")
    
    # 케이스 2: 가장 낮은 스코어 모델 제거
    print("\n" + "=" * 70)
    print(f"케이스 2: 2개 모델 앙상블 ({lowest_model} 제외)")
    print("=" * 70)
    
    exclude_for_case2 = ["DecisionTree", "NeuralNetwork", lowest_model]
    success2 = run_ensemble_case("case2", exclude_models=exclude_for_case2)
    if not success2:
        print("케이스 2 실행 실패")
        return
    
    case2_result = load_ensemble_result(results_dir, "case2")
    if case2_result is None:
        print("케이스 2 결과 파일을 찾을 수 없습니다.")
        return
    
    # 결과 비교
    print("\n" + "=" * 70)
    print("비교 결과")
    print("=" * 70)
    
    comparison_data = {
        "케이스 1 (3개 모델)": [
            ", ".join(case1_result['best_models'].keys()),
            case1_result['ensemble_result']['score'],
            case1_result['ensemble_result']['f1'],
            case1_result['ensemble_result']['auroc']
        ],
        "케이스 2 (2개 모델)": [
            ", ".join(case2_result['best_models'].keys()),
            case2_result['ensemble_result']['score'],
            case2_result['ensemble_result']['f1'],
            case2_result['ensemble_result']['auroc']
        ],
        "차이 (케이스 1 - 케이스 2)": [
            "",
            case1_result['ensemble_result']['score'] - case2_result['ensemble_result']['score'],
            case1_result['ensemble_result']['f1'] - case2_result['ensemble_result']['f1'],
            case1_result['ensemble_result']['auroc'] - case2_result['ensemble_result']['auroc']
        ]
    }
    
    comparison_df = pd.DataFrame(
        comparison_data,
        index=["모델", "Test Score", "F1-score", "AUROC"]
    )
    
    print("\n" + comparison_df.to_string())
    
    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)
    
    score_diff = case1_result['ensemble_result']['score'] - case2_result['ensemble_result']['score']
    f1_diff = case1_result['ensemble_result']['f1'] - case2_result['ensemble_result']['f1']
    auroc_diff = case1_result['ensemble_result']['auroc'] - case2_result['ensemble_result']['auroc']
    
    if score_diff > 0:
        print("✓ 케이스 1 (3개 모델)이 더 좋은 성능을 보입니다.")
        print(f"  성능 향상: {score_diff:.4f}")
        recommendation = "케이스 1 (3개 모델) 권장"
    else:
        print("✓ 케이스 2 (2개 모델)이 더 좋은 성능을 보입니다.")
        print(f"  성능 향상: {abs(score_diff):.4f}")
        recommendation = "케이스 2 (2개 모델) 권장"
    
    print(f"\n권장사항: {recommendation}")
    
    # 비교 결과 저장
    comparison_result = {
        "case1": {
            "models": list(case1_result['best_models'].keys()),
            "metrics": case1_result['ensemble_result'],
            "weights": case1_result.get('weights', {})
        },
        "case2": {
            "models": list(case2_result['best_models'].keys()),
            "removed_model": lowest_model,
            "metrics": case2_result['ensemble_result'],
            "weights": case2_result.get('weights', {})
        },
        "comparison": {
            "score_diff": float(score_diff),
            "f1_diff": float(f1_diff),
            "auroc_diff": float(auroc_diff),
            "recommendation": recommendation
        }
    }
    
    output_path = results_dir / "ensemble_cases_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    print(f"\n비교 결과 저장: {output_path}")
    
    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)


if __name__ == "__main__":
    compare_ensemble_cases()
