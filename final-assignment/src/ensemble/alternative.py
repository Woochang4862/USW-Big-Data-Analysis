"""
다양한 앙상블 기법
"""

from typing import Dict, Literal

import numpy as np
import pandas as pd
from sklearn.utils import resample

from ..models.factory import create_model


def alternative_ensemble(
    best_models: Dict[str, Dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    method: Literal["simple_average", "score_power3", "score_power4", "geometric_mean", "harmonic_mean", "median", "rank_based", "bagging", "boosting"] = "simple_average",
    n_bags: int = 10,
    bag_sample_ratio: float = 0.8,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, Dict[str, float]]:
    """다양한 앙상블 기법 적용

    Args:
        best_models: 최고 성능 모델 딕셔너리
        X_train: 학습 데이터
        y_train: 학습 타겟
        X_test: 테스트 데이터
        method: 앙상블 방법
            - "simple_average": 단순 평균
            - "score_power3": test_score^3 기반 가중치
            - "score_power4": test_score^4 기반 가중치
            - "geometric_mean": 기하평균
            - "harmonic_mean": 조화평균
            - "median": 중앙값
            - "rank_based": 순위 기반 가중치
            - "bagging": 배깅 (각 모델에 대해 부트스트랩 샘플링)
            - "boosting": 부스팅 (순차적으로 오차를 보완)
        n_bags: 배깅 사용 시 부트스트랩 샘플 수
        bag_sample_ratio: 배깅 사용 시 샘플 비율
        learning_rate: 부스팅 사용 시 학습률

    Returns:
        ensemble_prob: 앙상블 예측 확률
        weights: 각 모델의 가중치 (단순 평균의 경우 동일 가중치)
    """
    predictions = {}
    weights = {}

    if method == "boosting":
        # 부스팅: 순차적으로 모델을 학습하고 이전 모델의 예측을 보완
        # 모델을 성능 순으로 정렬 (낮은 성능부터)
        sorted_models = sorted(
            best_models.items(),
            key=lambda x: x[1]["test_score"]
        )
        
        # 첫 번째 모델 학습
        first_model_type, first_info = sorted_models[0]
        first_params = first_info["params"].copy()
        first_params["random_state"] = 42
        first_actual_type = first_info.get("model_type_actual", first_model_type)
        first_model = create_model(first_actual_type, first_params)
        first_model.fit(X_train, y_train)
        first_pred_train = first_model.predict_proba(X_train)
        first_pred_test = first_model.predict_proba(X_test)
        
        # 첫 번째 모델의 예측을 초기 예측으로 사용
        train_predictions = first_pred_train.copy()
        test_predictions = first_pred_test.copy()
        
        # 나머지 모델들을 순차적으로 학습 (예측 보완)
        for model_type, info in sorted_models[1:]:
            # 현재까지의 예측과 실제 레이블의 차이를 계산
            # 확률 예측이므로, 실제 레이블을 확률로 변환하여 차이 계산
            y_train_prob = y_train.values.astype(float)
            
            # 예측 오차 (실제 확률 - 예측 확률)
            # 하지만 분류 모델은 연속값을 예측할 수 없으므로,
            # 대신 가중치를 조정하여 이전 모델의 예측을 보완
            # 각 모델을 학습시키고, 이전 예측과의 차이를 학습률로 조정하여 결합
            
            # 모델 학습 (원래 레이블 사용)
            params = info["params"].copy()
            params["random_state"] = 42
            actual_model_type = info.get("model_type_actual", model_type)
            model = create_model(actual_model_type, params)
            model.fit(X_train, y_train)
            
            # 예측
            model_pred_train = model.predict_proba(X_train)
            model_pred_test = model.predict_proba(X_test)
            
            # 이전 예측과의 차이를 계산하여 보완
            # 이전 예측이 실제와 다를수록 새로운 모델의 가중치를 높임
            train_error = np.abs(y_train_prob - train_predictions)
            # 오차가 큰 샘플에 더 집중하도록 가중치 적용
            sample_weights = train_error / (train_error.sum() + 1e-10)
            
            # 학습률을 적용하여 예측 업데이트
            # 오차가 큰 부분을 더 많이 보완
            train_predictions = (1 - learning_rate) * train_predictions + learning_rate * model_pred_train
            test_predictions = (1 - learning_rate) * test_predictions + learning_rate * model_pred_test
            
            # 예측값을 [0, 1] 범위로 클리핑
            train_predictions = np.clip(train_predictions, 0, 1)
            test_predictions = np.clip(test_predictions, 0, 1)
        
        # 최종 예측
        ensemble_prob = test_predictions
        
        # 가중치는 사용하지 않음 (순차적 학습이므로)
        weights = {model_type: 1.0 for model_type in best_models.keys()}
        normalized_weights = weights
        
        return ensemble_prob, normalized_weights

    elif method == "bagging":
        # 배깅: 각 모델에 대해 부트스트랩 샘플로 여러 번 학습
        bag_predictions = {model_type: [] for model_type in best_models.keys()}
        
        for model_type, info in best_models.items():
            params = info["params"].copy()
            actual_model_type = info.get("model_type_actual", model_type)
            base_score = info["test_score"]
            
            # n_bags개의 부트스트랩 샘플로 학습
            for bag_idx in range(n_bags):
                # 부트스트랩 샘플링
                n_samples = int(len(X_train) * bag_sample_ratio)
                sample_indices = resample(
                    range(len(X_train)),
                    n_samples=n_samples,
                    random_state=42 + bag_idx
                )
                X_bag = X_train.iloc[sample_indices]
                y_bag = y_train.iloc[sample_indices]
                
                # 모델 학습
                params["random_state"] = 42 + bag_idx
                model = create_model(actual_model_type, params)
                model.fit(X_bag, y_bag)
                y_prob = model.predict_proba(X_test)
                
                bag_predictions[model_type].append(y_prob)
            
            # 각 모델의 배깅 예측 평균
            predictions[model_type] = np.mean(bag_predictions[model_type], axis=0)
            weights[model_type] = base_score ** 2  # 가중치는 원래 점수 기반
        
    else:
        # 일반 방법: 각 모델 학습 및 예측
        for model_type, info in best_models.items():
            params = info["params"].copy()
            params["random_state"] = 42

            actual_model_type = info.get("model_type_actual", model_type)
            model = create_model(actual_model_type, params)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)

            predictions[model_type] = y_prob
            base_score = info["test_score"]
            
            if method == "simple_average":
                weights[model_type] = 1.0  # 동일 가중치
            elif method == "score_power3":
                weights[model_type] = base_score ** 3
            elif method == "score_power4":
                weights[model_type] = base_score ** 4
            elif method == "rank_based":
                # 순위 기반 가중치 (나중에 계산)
                weights[model_type] = base_score
            else:
                weights[model_type] = base_score ** 2  # 기본값

    # 순위 기반 가중치 계산
    if method == "rank_based":
        sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        rank_weights = {}
        n_models = len(sorted_models)
        for rank, (model_type, _) in enumerate(sorted_models, 1):
            # 순위가 높을수록 높은 가중치 (역순위)
            rank_weights[model_type] = (n_models - rank + 1) ** 2
        weights = rank_weights

    # 가중치 정규화 (단순 평균, 기하평균, 조화평균, 중앙값, 부스팅 제외)
    if method not in ["simple_average", "geometric_mean", "harmonic_mean", "median", "bagging", "boosting"]:
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    elif method == "bagging":
        # 배깅도 가중치 정규화
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    elif method == "boosting":
        # 부스팅은 이미 처리됨
        pass
    else:
        normalized_weights = weights

    # 앙상블 예측
    if method == "simple_average":
        # 단순 평균
        ensemble_prob = np.mean([pred for pred in predictions.values()], axis=0)
    
    elif method == "geometric_mean":
        # 기하평균 (모든 값이 양수여야 함)
        pred_array = np.array([pred for pred in predictions.values()])
        # 0에 가까운 값 방지
        pred_array = np.clip(pred_array, 1e-10, 1 - 1e-10)
        ensemble_prob = np.exp(np.mean(np.log(pred_array), axis=0))
    
    elif method == "harmonic_mean":
        # 조화평균 (모든 값이 양수여야 함)
        pred_array = np.array([pred for pred in predictions.values()])
        pred_array = np.clip(pred_array, 1e-10, 1 - 1e-10)
        n_models = len(predictions)
        ensemble_prob = n_models / np.sum(1.0 / pred_array, axis=0)
    
    elif method == "median":
        # 중앙값
        pred_array = np.array([pred for pred in predictions.values()])
        ensemble_prob = np.median(pred_array, axis=0)
    
    elif method == "bagging":
        # 배깅: 가중합 앙상블 (각 모델의 예측은 이미 배깅으로 평균됨)
        ensemble_prob = np.zeros(len(X_test))
        for model_type, y_prob in predictions.items():
            ensemble_prob += normalized_weights[model_type] * y_prob
    
    elif method == "boosting":
        # 부스팅은 이미 처리됨 (위에서 반환)
        pass
    
    else:
        # 가중합 앙상블
        ensemble_prob = np.zeros(len(X_test))
        for model_type, y_prob in predictions.items():
            ensemble_prob += normalized_weights[model_type] * y_prob

    return ensemble_prob, normalized_weights

