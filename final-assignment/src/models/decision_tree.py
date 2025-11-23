"""
Decision Tree 모델
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..data.preprocessing import encode_categorical_features
from .base import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree 모델"""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "DecisionTreeModel":
        """모델 학습"""
        X_train_enc, _ = encode_categorical_features(X_train, X_train)

        self.model = DecisionTreeClassifier(
            max_depth=self.params.get("max_depth"),
            random_state=self.params.get("random_state", 42),
        )
        self.model.fit(X_train_enc, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """예측 확률 반환"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_enc, _ = encode_categorical_features(X, X)
        return self.model.predict_proba(X_enc)[:, 1]

