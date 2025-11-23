"""
가중합 앙상블
"""

from typing import Dict

import numpy as np
import pandas as pd

from ..models.factory import create_model


def weighted_ensemble(
    best_models: Dict[str, Dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """가중합 앙상블 예측

    Args:
        best_models: 최고 성능 모델 딕셔너리
        X_train: 학습 데이터
        y_train: 학습 타겟
        X_test: 테스트 데이터

    Returns:
        ensemble_prob: 앙상블 예측 확률
        weights: 각 모델의 가중치
    """
    predictions = {}
    weights = {}

    # 각 모델 학습 및 예측
    for model_type, info in best_models.items():
        params = info["params"].copy()
        params["random_state"] = 42

        actual_model_type = info.get("model_type_actual", model_type)
        model = create_model(actual_model_type, params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        predictions[model_type] = y_prob
        # test_score를 제곱하여 가중치로 사용
        base_score = info["test_score"]
        weights[model_type] = base_score ** 2

    # 가중치 정규화
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # 가중합 앙상블
    ensemble_prob = np.zeros(len(X_test))
    for model_type, y_prob in predictions.items():
        ensemble_prob += normalized_weights[model_type] * y_prob

    return ensemble_prob, normalized_weights

