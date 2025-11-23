"""
기본 모델 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """모델 기본 클래스"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.scaler = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        """모델 학습"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """예측 확률 반환"""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 클래스 반환"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

