"""
Logistic Regression 모델
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.preprocessing import encode_categorical_features
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression 모델"""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogisticRegressionModel":
        """모델 학습"""
        X_train_enc, _ = encode_categorical_features(X_train, X_train)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_enc)

        self.model = LogisticRegression(
            C=self.params.get("C", 1.0),
            max_iter=self.params.get("max_iter", 1000),
            random_state=self.params.get("random_state", 42),
        )
        self.model.fit(X_train_scaled, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """예측 확률 반환"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_enc, _ = encode_categorical_features(X, X)
        X_scaled = self.scaler.transform(X_enc)
        return self.model.predict_proba(X_scaled)[:, 1]

