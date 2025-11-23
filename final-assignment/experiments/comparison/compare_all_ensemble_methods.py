#!/usr/bin/env python
"""
모든 앙상블 방법 비교
- 케이스 1: 3개 모델 가중합
- 케이스 2: 2개 모델 가중합 (가장 낮은 스코어 모델 제거)
- 케이스 3: 3개 모델 스태킹
JSON 결과를 로드하여 비교
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def load_ensemble_result(results_dir: Path, case_name: str) -> dict:
    """앙상블 결과 JSON 로드"""
    result_path = results_dir / f"ensemble_results_{case_name}.json"
    if not result_path.exists():
        return None

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_all_methods():
    """모든 앙상블 방법 비교"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = base_dir / "results"

    print("=" * 70)
    print("모든 앙상블 방법 비교")
    print("=" * 70)

    # 결과 로드
    case1_result = load_ensemble_result(results_dir, "case1")
    case2_result = load_ensemble_result(results_dir, "case2")
    stacking_result = load_ensemble_result(results_dir, "stacking")

    if not all([case1_result, case2_result, stacking_result]):
        print("일부 결과 파일이 없습니다. 모든 케이스를 실행해주세요.")
        return

    # 비교 테이블 생성
    comparison_data = {
        "케이스 1 (3개 가중합)": [
            ", ".join(case1_result["best_models"].keys()),
            case1_result["ensemble_result"]["score"],
            case1_result["ensemble_result"]["f1"],
            case1_result["ensemble_result"]["auroc"],
            case1_result["ensemble_method"],
        ],
        "케이스 2 (2개 가중합)": [
            ", ".join(case2_result["best_models"].keys()),
            case2_result["ensemble_result"]["score"],
            case2_result["ensemble_result"]["f1"],
            case2_result["ensemble_result"]["auroc"],
            case2_result["ensemble_method"],
        ],
        "케이스 3 (3개 스태킹)": [
            ", ".join(stacking_result["best_models"].keys()),
            stacking_result["ensemble_result"]["score"],
            stacking_result["ensemble_result"]["f1"],
            stacking_result["ensemble_result"]["auroc"],
            stacking_result["ensemble_method"],
        ],
    }

    comparison_df = pd.DataFrame(
        comparison_data, index=["모델", "Test Score", "F1-score", "AUROC", "방법"]
    )

    print("\n" + comparison_df.to_string())

    # 최고 성능 찾기
    print("\n" + "=" * 70)
    print("성능 순위")
    print("=" * 70)

    scores = {
        "케이스 1 (3개 가중합)": case1_result["ensemble_result"]["score"],
        "케이스 2 (2개 가중합)": case2_result["ensemble_result"]["score"],
        "케이스 3 (3개 스태킹)": stacking_result["ensemble_result"]["score"],
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (method, score) in enumerate(sorted_scores, 1):
        print(f"{rank}. {method}: {score:.4f}")

    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    best_method, best_score = sorted_scores[0]
    print(f"✓ 최고 성능: {best_method}")
    print(f"  Test Score: {best_score:.4f}")

    # 상세 비교
    print("\n상세 비교:")
    print(
        f"  케이스 1 vs 케이스 2: "
        f"{case1_result['ensemble_result']['score'] - case2_result['ensemble_result']['score']:+.4f}"
    )
    print(
        f"  케이스 1 vs 케이스 3: "
        f"{case1_result['ensemble_result']['score'] - stacking_result['ensemble_result']['score']:+.4f}"
    )
    print(
        f"  케이스 2 vs 케이스 3: "
        f"{case2_result['ensemble_result']['score'] - stacking_result['ensemble_result']['score']:+.4f}"
    )

    # 결과 저장
    comparison_result = {
        "case1_weighted_3models": {
            "models": list(case1_result["best_models"].keys()),
            "metrics": case1_result["ensemble_result"],
            "method": case1_result["ensemble_method"],
        },
        "case2_weighted_2models": {
            "models": list(case2_result["best_models"].keys()),
            "metrics": case2_result["ensemble_result"],
            "method": case2_result["ensemble_method"],
        },
        "case3_stacking_3models": {
            "models": list(stacking_result["best_models"].keys()),
            "metrics": stacking_result["ensemble_result"],
            "method": stacking_result["ensemble_method"],
        },
        "ranking": [
            {"rank": 1, "method": sorted_scores[0][0], "score": float(sorted_scores[0][1])},
            {"rank": 2, "method": sorted_scores[1][0], "score": float(sorted_scores[1][1])},
            {"rank": 3, "method": sorted_scores[2][0], "score": float(sorted_scores[2][1])},
        ],
        "best_method": best_method,
    }

    output_path = results_dir / "all_ensemble_methods_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    print(f"\n비교 결과 저장: {output_path}")

    print("\n" + "=" * 70)
    print("비교 완료!")
    print("=" * 70)


if __name__ == "__main__":
    compare_all_methods()
