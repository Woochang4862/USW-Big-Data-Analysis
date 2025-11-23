"""
스태킹 앙상블
"""

from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ..models.factory import create_model


def stacking_ensemble(
    best_models: Dict[str, Dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_folds: int = 5,
    meta_C: float = 1.0,
    meta_model_type: str = "LogisticRegression",
    meta_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Any, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """스태킹 앙상블 구현

    Args:
        best_models: 최고 성능 모델 딕셔너리
        X_train: 학습 데이터
        y_train: 학습 타겟
        X_test: 테스트 데이터
        n_folds: Cross-validation 폴드 수
        meta_C: 메타 모델(LogisticRegression)의 C 파라미터 (복잡도 조절, LogisticRegression 사용 시)
        meta_model_type: 메타 모델 타입 ("LogisticRegression", "RandomForest", 등)
        meta_params: 메타 모델 파라미터 딕셔너리 (meta_model_type에 따라 사용)

    Returns:
        ensemble_prob: 앙상블 예측 확률
        meta_model: 메타 모델
        oof_predictions: OOF 예측 (각 모델별)
        test_predictions: 테스트 예측 (각 모델별)
    """
    # Cross-validation 설정
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # OOF 예측 저장소
    oof_predictions = {
        model_type: np.zeros(len(X_train)) for model_type in best_models.keys()
    }
    test_predictions = {model_type: [] for model_type in best_models.keys()}

    # Base models 학습 및 OOF 예측 생성
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_test_preds = {}

        for model_type, info in best_models.items():
            params = info["params"].copy()
            params["random_state"] = 42

            actual_model_type = info.get("model_type_actual", model_type)

            # 모델 학습
            model = create_model(actual_model_type, params)
            model.fit(X_tr, y_tr)

            # Validation 예측
            oof_pred = model.predict_proba(X_val)
            oof_predictions[model_type][val_idx] = oof_pred

            # Test 예측
            test_pred = model.predict_proba(X_test)
            fold_test_preds[model_type] = test_pred

        # Fold별 테스트 예측 저장
        for model_type, pred in fold_test_preds.items():
            test_predictions[model_type].append(pred)

    # 테스트 예측 평균 (각 모델별)
    for model_type in test_predictions:
        test_predictions[model_type] = np.mean(test_predictions[model_type], axis=0)

    # 메타 피처 생성 (OOF 예측들을 결합)
    meta_X_train = np.column_stack([oof_predictions[mt] for mt in best_models.keys()])
    meta_X_test = np.column_stack([test_predictions[mt] for mt in best_models.keys()])

    # 메타 모델 생성
    if meta_model_type == "LogisticRegression":
        meta_params_final = {"C": meta_C, "max_iter": 1000, "random_state": 42}
        if meta_params:
            meta_params_final.update(meta_params)
        meta_model = LogisticRegression(**meta_params_final)
        # numpy array를 그대로 사용
        meta_X_train_fit = meta_X_train
        meta_X_test_fit = meta_X_test
    elif meta_model_type == "RandomForest":
        # RandomForest는 직접 sklearn 사용 (범주형 변수 처리 없이)
        from sklearn.ensemble import RandomForestClassifier
        if meta_params is None:
            meta_params = {}
        meta_params["random_state"] = 42
        meta_model = RandomForestClassifier(**meta_params)
        # numpy array를 그대로 사용 (sklearn은 numpy array를 받음)
        meta_X_train_fit = meta_X_train
        meta_X_test_fit = meta_X_test
    else:
        # 다른 메타 모델 타입 (XGBoost 등)
        if meta_params is None:
            meta_params = {}
        meta_params["random_state"] = 42
        meta_model = create_model(meta_model_type, meta_params)
        # pandas DataFrame으로 변환 (일부 모델이 DataFrame을 기대할 수 있음)
        meta_X_train_fit = pd.DataFrame(meta_X_train, columns=[f"base_{i}" for i in range(meta_X_train.shape[1])])
        meta_X_test_fit = pd.DataFrame(meta_X_test, columns=[f"base_{i}" for i in range(meta_X_test.shape[1])])
    
    meta_model.fit(meta_X_train_fit, y_train)

    # 최종 예측
    meta_proba = meta_model.predict_proba(meta_X_test_fit)
    # predict_proba가 2차원 배열인지 확인
    if meta_proba.ndim == 2 and meta_proba.shape[1] > 1:
        ensemble_prob = meta_proba[:, 1]
    else:
        # 1차원 배열이거나 단일 클래스인 경우
        ensemble_prob = meta_proba if meta_proba.ndim == 1 else meta_proba.flatten()

    return ensemble_prob, meta_model, oof_predictions, test_predictions

