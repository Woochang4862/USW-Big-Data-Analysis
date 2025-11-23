#!/usr/bin/env python
"""
스태킹 vs 가중합 앙상블 비교
- 동일한 모델 종류와 개수로 스태킹과 가중합을 비교
- JSON 결과를 로드하여 비교
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_ensemble_result(results_dir: Path, case_name: str) -> dict:
    """앙상블 결과 JSON 로드"""
    result_path = results_dir / f"ensemble_results_{case_name}.json"
    if not result_path.exists():
        return None
    
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_stacking_vs_weighted():
    """스태킹 vs 가중합 비교"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    
    print("=" * 70)
    print("스태킹 vs 가중합 앙상블 비교 (동일한 모델로)")
    print("=" * 70)
    
    # 비교 케이스 정의
    comparison_cases = [
        {
            "name": "3개 모델 비교",
            "weighted_case": "weighted_3models",
            "stacking_case": "stacking_3models",
            "models": ["LogisticRegression", "RandomForest", "XGBoost"]
        },
        {
            "name": "2개 모델 비교",
            "weighted_case": "weighted_2models",
            "stacking_case": "stacking_2models",
            "models": ["LogisticRegression", "XGBoost"]
        }
    ]
    
    all_results = {}
    
    for case_info in comparison_cases:
        case_name = case_info["name"]
        
        print("\n" + "=" * 70)
        print(f"{case_name}")
        print("=" * 70)
        
        # 결과 로드
        weighted_result = load_ensemble_result(results_dir, case_info["weighted_case"])
        stacking_result = load_ensemble_result(results_dir, case_info["stacking_case"])
        
        if weighted_result is None or stacking_result is None:
            print(f"결과 파일이 없습니다. weighted: {weighted_result is not None}, stacking: {stacking_result is not None}")
            continue
        
        all_results[case_name] = {
            "weighted": weighted_result,
            "stacking": stacking_result,
            "models": case_info["models"]
        }
        
        print(f"\n모델: {', '.join(case_info['models'])}")
        print(f"\n가중합:")
        print(f"  Test Score: {weighted_result['ensemble_result']['score']:.4f}")
        print(f"  F1-score: {weighted_result['ensemble_result']['f1']:.4f}")
        print(f"  AUROC: {weighted_result['ensemble_result']['auroc']:.4f}")
        print(f"\n스태킹:")
        print(f"  Test Score: {stacking_result['ensemble_result']['score']:.4f}")
        print(f"  F1-score: {stacking_result['ensemble_result']['f1']:.4f}")
        print(f"  AUROC: {stacking_result['ensemble_result']['auroc']:.4f}")
        
        score_diff = stacking_result['ensemble_result']['score'] - weighted_result['ensemble_result']['score']
        f1_diff = stacking_result['ensemble_result']['f1'] - weighted_result['ensemble_result']['f1']
        auroc_diff = stacking_result['ensemble_result']['auroc'] - weighted_result['ensemble_result']['auroc']
        
        print(f"\n차이 (스태킹 - 가중합):")
        print(f"  Test Score: {score_diff:+.4f}")
        print(f"  F1-score: {f1_diff:+.4f}")
        print(f"  AUROC: {auroc_diff:+.4f}")
        
        if score_diff > 0:
            print(f"\n✓ 스태킹이 더 좋습니다 (차이: +{score_diff:.4f})")
        else:
            print(f"\n✓ 가중합이 더 좋습니다 (차이: {score_diff:.4f})")
    
    # 전체 비교 테이블
    print("\n" + "=" * 70)
    print("전체 비교 요약")
    print("=" * 70)
    
    comparison_data = []
    
    for case_name, results in all_results.items():
        weighted = results["weighted"]
        stacking = results["stacking"]
        
        score_diff = stacking['ensemble_result']['score'] - weighted['ensemble_result']['score']
        
        comparison_data.append({
            "케이스": case_name,
            "모델": ", ".join(results["models"]),
            "가중합 Score": weighted['ensemble_result']['score'],
            "스태킹 Score": stacking['ensemble_result']['score'],
            "차이": score_diff,
            "승자": "스태킹" if score_diff > 0 else "가중합"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)
    
    for case_name, results in all_results.items():
        weighted = results["weighted"]
        stacking = results["stacking"]
        score_diff = stacking['ensemble_result']['score'] - weighted['ensemble_result']['score']
        
        print(f"\n{case_name}:")
        if score_diff > 0:
            print(f"  스태킹이 더 좋습니다 (차이: +{score_diff:.4f})")
        else:
            print(f"  가중합이 더 좋습니다 (차이: {score_diff:.4f})")
    
    # 결과 저장
    comparison_result = {
        "comparisons": []
    }
    
    for case_name, results in all_results.items():
        weighted = results["weighted"]
        stacking = results["stacking"]
        
        score_diff = stacking['ensemble_result']['score'] - weighted['ensemble_result']['score']
        
        comparison_result["comparisons"].append({
            "case": case_name,
            "models": results["models"],
            "weighted": {
                "metrics": weighted['ensemble_result'],
                "method": weighted['ensemble_method']
            },
            "stacking": {
                "metrics": stacking['ensemble_result'],
                "method": stacking['ensemble_method']
            },
            "difference": {
                "score": float(score_diff),
                "f1": float(stacking['ensemble_result']['f1'] - weighted['ensemble_result']['f1']),
                "auroc": float(stacking['ensemble_result']['auroc'] - weighted['ensemble_result']['auroc'])
            },
            "winner": "stacking" if score_diff > 0 else "weighted"
        })
    
    output_path = results_dir / "stacking_vs_weighted_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    print(f"\n비교 결과 저장: {output_path}")
    
    print("\n" + "=" * 70)
    print("비교 완료!")
    print("=" * 70)


if __name__ == "__main__":
    compare_stacking_vs_weighted()
