"""
모델 팩토리
"""

from typing import Any, Dict

import pandas as pd

from .base import BaseModel
from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel

try:
    from .neural_network import NeuralNetworkModel

    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False


class ModelFactory:
    """모델 생성 팩토리"""

    _model_classes = {
        "LogisticRegression": LogisticRegressionModel,
        "DecisionTree": DecisionTreeModel,
        "RandomForest": RandomForestModel,
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel,
    }

    if NN_AVAILABLE:
        _model_classes["NeuralNetwork"] = NeuralNetworkModel

    @classmethod
    def create(cls, model_type: str, params: Dict[str, Any]) -> BaseModel:
        """모델 생성"""
        if model_type not in cls._model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._model_classes[model_type](params)

    @classmethod
    def get_available_models(cls):
        """사용 가능한 모델 목록 반환"""
        return list(cls._model_classes.keys())


def create_model(model_type: str, params: Dict[str, Any]) -> BaseModel:
    """모델 생성 헬퍼 함수"""
    return ModelFactory.create(model_type, params)

