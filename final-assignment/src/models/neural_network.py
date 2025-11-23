"""
Neural Network 모델
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..data.preprocessing import encode_categorical_features
from .base import BaseModel


class NeuralNetworkModel(BaseModel):
    """Neural Network 모델"""

    def __init__(self, params: Dict[str, Any]):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        super().__init__(params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NeuralNetworkModel":
        """모델 학습"""
        X_train_enc, _ = encode_categorical_features(X_train, X_train)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_enc)

        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train_scaled.shape[1],)))

        layers_config = self.params.get("layers", [64])
        for units in layers_config:
            model.add(layers.Dense(units, activation="relu"))

        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        epochs = self.params.get("epochs", 50)
        model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=self.params.get("batch_size", 64),
            verbose=0,
            validation_split=0.1,
        )

        self.model = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """예측 확률 반환"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_enc, _ = encode_categorical_features(X, X)
        X_scaled = self.scaler.transform(X_enc)
        return self.model.predict(X_scaled, verbose=0).flatten()

