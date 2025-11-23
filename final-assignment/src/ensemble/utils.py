"""
앙상블 유틸리티
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_results(results_path: Path) -> List[Dict]:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model_info(result: Dict) -> Dict:
    return {
        "config_name": result["config_name"],
        "params": result["params"],
        "test_score": result["test_score"],
        "test_f1": result["test_f1"],
        "test_auroc": result["test_auroc"],
        "model_type_actual": result["model_type"],
    }


def find_best_models(
    results_path: Path,
    exclude_models: Optional[List[str]] = None,
    force_catboost_level1: bool = False,
) -> Dict[str, Dict]:
    """각 모델 타입별 최고 test_score를 가진 설정 찾기"""

    results = _load_results(results_path)

    # 제외할 모델 타입
    if exclude_models is None:
        exclude_models = ["NeuralNetwork", "DecisionTree"]

    best_models = {}
    catboost_level1 = None

    # CatBoost Level1 강제 사용 옵션
    if force_catboost_level1:
        for result in results:
            if result["model_type"] == "CatBoost" and result["config_name"] == "Level1":
                catboost_level1 = _build_model_info(result)
                break

        if catboost_level1 is None:
            raise ValueError(
                "CatBoost Level1을 찾을 수 없습니다. "
                "overfitting_experiments.json에 CatBoost Level1 결과가 있는지 확인하세요."
            )

    for result in results:
        model_type = result["model_type"]

        if model_type in exclude_models:
            continue

        # CatBoost Level1 강제 사용 옵션이 있으면 CatBoost는 스킵
        if force_catboost_level1 and model_type == "CatBoost":
            continue

        test_score = result["test_score"]

        if model_type not in best_models or test_score > best_models[model_type]["test_score"]:
            best_models[model_type] = _build_model_info(result)

    # CatBoost Level1 강제 사용 옵션이 있으면 추가
    if force_catboost_level1 and catboost_level1:
        best_models["CatBoost"] = catboost_level1

    return best_models


def get_model_by_rank(
    results_path: Path,
    model_type: str,
    rank: int = 2,
) -> Optional[Dict]:
    """특정 모델 타입에서 test_score 기준 rank 번째 설정 반환"""

    results = _load_results(results_path)
    candidates = [r for r in results if r["model_type"] == model_type]

    if not candidates or rank < 1 or rank > len(candidates):
        return None

    candidates.sort(key=lambda x: x["test_score"], reverse=True)
    target = candidates[rank - 1]
    return _build_model_info(target)

