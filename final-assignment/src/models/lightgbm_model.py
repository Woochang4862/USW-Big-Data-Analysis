"""
LightGBM 모델
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

from ..data.preprocessing import infer_categorical_from_dtype, impute_by_rules
from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM 모델 - 범주형 변수를 직접 활용"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.categorical_encoders = {}  # 범주형 변수 인코더 저장

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMModel":
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
        
        # LightGBM은 범주형 변수가 0 이상의 정수여야 함 (음수 값 경고 방지)
        # 각 범주형 변수를 0부터 시작하는 연속된 정수로 재인코딩
        self.categorical_encoders = {}
        for col in categorical_cols:
            if col in X_train_imp.columns:
                # float인 범주형 변수를 먼저 정수로 변환
                if pd.api.types.is_float_dtype(X_train_imp[col]):
                    X_train_imp[col] = X_train_imp[col].apply(
                        lambda x: int(x) if pd.notna(x) else x
                    )
                
                # 고유한 값들을 0부터 시작하는 정수로 매핑 (NaN 제외)
                unique_values = X_train_imp[col].dropna().unique()
                # 정렬하여 일관된 매핑 보장
                unique_values_sorted = sorted(unique_values)
                value_to_code = {val: idx for idx, val in enumerate(unique_values_sorted)}
                
                # 인코더 저장 (예측 시 사용)
                self.categorical_encoders[col] = value_to_code
                
                # 값 재인코딩 (NaN은 그대로 유지)
                X_train_imp[col] = X_train_imp[col].map(
                    lambda x: value_to_code.get(x, np.nan) if pd.notna(x) else np.nan
                )

        # LightGBMClassifier 파라미터 준비
        self.model = lgb.LGBMClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.1),
            num_leaves=self.params.get("num_leaves", 31),
            random_state=self.params.get("random_state", 42),
            n_jobs=self.params.get("n_jobs", -1),
            verbose=-1,
        )
        
        # 범주형 변수가 있을 때만 categorical_feature 전달
        if cat_indices:
            self.model.fit(X_train_imp, y_train, categorical_feature=cat_indices)
        else:
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
        
        # LightGBM은 범주형 변수가 0 이상의 정수여야 함
        # fit에서 사용한 동일한 인코더 적용
        for col in categorical_cols:
            if col in X_imp.columns:
                # float인 범주형 변수를 먼저 정수로 변환
                if pd.api.types.is_float_dtype(X_imp[col]):
                    X_imp[col] = X_imp[col].apply(
                        lambda x: int(x) if pd.notna(x) else x
                    )
                
                # fit에서 저장한 인코더 사용
                if col in self.categorical_encoders:
                    value_to_code = self.categorical_encoders[col]
                    # 알려지지 않은 값은 NaN으로 처리
                    X_imp[col] = X_imp[col].map(
                        lambda x: value_to_code.get(x, np.nan) if pd.notna(x) else np.nan
                    )
        
        return self.model.predict_proba(X_imp)[:, 1]


