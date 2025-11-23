"""
CatBoost 모델
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..data.preprocessing import infer_categorical_from_dtype, impute_by_rules
from .base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost 모델 - 범주형 변수를 직접 활용"""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostModel":
        """모델 학습 - 범주형 변수를 그대로 사용"""
        # 범주형 변수 감지 (메타데이터 포함)
        categorical_cols = infer_categorical_from_dtype(X_train, use_metadata=True)
        
        # 범주형 변수가 있는 경우에만 인덱스 추출
        cat_indices = []
        if categorical_cols:
            cat_indices = [
                X_train.columns.get_loc(col)
                for col in categorical_cols
                if col in X_train.columns
            ]

        # 결측치 처리만 수행 (인코딩은 하지 않음)
        X_train_imp, _ = impute_by_rules(X_train, X_train, categorical_cols)
        
        # CatBoost는 범주형 변수가 정수나 문자열이어야 함
        # float인 범주형 변수를 정수로 변환
        for col in categorical_cols:
            if col in X_train_imp.columns:
                if pd.api.types.is_float_dtype(X_train_imp[col]):
                    # NaN은 그대로 두고, 정수 값만 변환
                    X_train_imp[col] = X_train_imp[col].apply(
                        lambda x: int(x) if pd.notna(x) else x
                    )

        # CatBoostClassifier 파라미터 준비
        model_params = {
            "iterations": self.params.get("iterations", 100),
            "depth": self.params.get("depth", 6),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "l2_leaf_reg": self.params.get("l2_leaf_reg", 3.0),
            "random_state": self.params.get("random_state", 42),
            "verbose": False,
        }
        
        # 범주형 변수가 있을 때만 cat_features 추가
        if cat_indices:
            model_params["cat_features"] = cat_indices

        self.model = CatBoostClassifier(**model_params)
        self.model.fit(X_train_imp, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """예측 확률 반환"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # 범주형 변수 감지 (fit과 동일한 로직)
        categorical_cols = infer_categorical_from_dtype(X, use_metadata=True)
        
        # 결측치 처리만 수행 (인코딩은 하지 않음)
        X_imp, _ = impute_by_rules(X, X, categorical_cols)
        
        # CatBoost는 범주형 변수가 정수나 문자열이어야 함
        # float인 범주형 변수를 정수로 변환 (fit과 동일)
        for col in categorical_cols:
            if col in X_imp.columns:
                if pd.api.types.is_float_dtype(X_imp[col]):
                    # NaN은 그대로 두고, 정수 값만 변환
                    X_imp[col] = X_imp[col].apply(
                        lambda x: int(x) if pd.notna(x) else x
                    )
        
        return self.model.predict_proba(X_imp)[:, 1]

