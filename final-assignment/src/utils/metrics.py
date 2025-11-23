"""
평가 메트릭 유틸리티
"""

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def evaluate_model(y_true, y_prob: np.ndarray) -> Dict[str, float]:
    """모델 평가: F1-score, AUROC"""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
    }


def calc_score(metric: Dict[str, float]) -> float:
    """score = (f1 + auroc) / 2"""
    return float((metric["f1"] + metric["auroc"]) / 2)


def build_result(
    model_type: str,
    config_name: str,
    params: Dict,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> Dict:
    """실험 결과 딕셔너리 생성"""
    train_score = calc_score(train_metrics)
    test_score = calc_score(test_metrics)
    return {
        "model_type": model_type,
        "config_name": config_name,
        "params": params,
        "train_f1": train_metrics["f1"],
        "train_auroc": train_metrics["auroc"],
        "train_score": train_score,
        "test_f1": test_metrics["f1"],
        "test_auroc": test_metrics["auroc"],
        "test_score": test_score,
        "overfitting_gap_f1": train_metrics["f1"] - test_metrics["f1"],
        "overfitting_gap_auroc": train_metrics["auroc"] - test_metrics["auroc"],
        "overfitting_gap_score": train_score - test_score,
    }

