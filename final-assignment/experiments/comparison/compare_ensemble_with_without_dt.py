#!/usr/bin/env python
"""
Decision Tree 포함 여부에 따른 앙상블 성능 비교 실험
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# ensemble_best_models.py의 함수들 재사용
def load_feature_label_pairs(raw_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """raw 데이터 로드"""
    X_train = pd.read_csv(raw_dir / "X_train.csv", index_col="ID")
    y_train = pd.read_csv(raw_dir / "Y_train.csv", index_col="ID")["HE_D3_label"]
    X_test = pd.read_csv(raw_dir / "X_test.csv", index_col="ID")
    y_test = pd.read_csv(raw_dir / "Y_test.csv", index_col="ID")["HE_D3_label"]
    return X_train, y_train, X_test, y_test


def infer_categorical_from_dtype(df: pd.DataFrame) -> List[str]:
    """dtype 기반 범주형 변수 추론"""
    from pandas.api.types import CategoricalDtype
    
    categorical_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, CategoricalDtype):
            categorical_cols.append(col)
    return categorical_cols


def impute_by_rules(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """결측치 처리: 범주형(최빈값), 수치형(중앙값)"""
    train_df = X_train.copy()
    test_df = X_test.copy()
    
    categorical_cols = list(categorical_cols)
    
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        mode_value = train_df[col].mode(dropna=True)
        if len(mode_value) == 0:
            continue
        value = mode_value.iloc[0]
        train_df[col] = train_df[col].fillna(value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(value)
    
    numeric_candidates = [c for c in train_df.columns if c not in categorical_cols]
    numeric_cols = [c for c in numeric_candidates if pd.api.types.is_numeric_dtype(train_df[c])]
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
    
    return train_df, test_df


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """범주형 변수를 원-핫 인코딩"""
    categorical_cols = infer_categorical_from_dtype(X_train)
    
    if categorical_cols:
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=False)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=False)
        
        common_cols = X_train_encoded.columns.intersection(X_test_encoded.columns)
        X_train_encoded = X_train_encoded[common_cols]
        X_test_encoded = X_test_encoded[common_cols]
        
        return X_train_encoded, X_test_encoded
    
    return X_train, X_test


def evaluate_model(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    """모델 평가: F1-score, AUROC"""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "score": float((f1_score(y_true, y_pred) + roc_auc_score(y_true, y_prob)) / 2),
    }


def find_best_models(results_path: Path, exclude_dt: bool = False) -> Dict[str, Dict]:
    """각 모델 타입별 최고 test_score를 가진 설정 찾기"""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    excluded_models = ["NeuralNetwork"]
    if exclude_dt:
        excluded_models.append("DecisionTree")
    
    best_models = {}
    
    for result in results:
        model_type = result["model_type"]
        
        if model_type in excluded_models:
            continue
        
        test_score = result["test_score"]
        
        if model_type not in best_models or test_score > best_models[model_type]["test_score"]:
            best_models[model_type] = {
                "config_name": result["config_name"],
                "params": result["params"],
                "test_score": test_score,
                "test_f1": result["test_f1"],
                "test_auroc": result["test_auroc"],
            }
    
    return best_models


def train_logistic_regression(X_train, y_train, params):
    """Logistic Regression 모델 학습"""
    X_train_enc, _ = encode_categorical_features(X_train, X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    
    model = LogisticRegression(C=params["C"], max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def predict_logistic_regression(model, scaler, X_test):
    """Logistic Regression 예측"""
    X_test_enc, _ = encode_categorical_features(X_test, X_test)
    X_test_scaled = scaler.transform(X_test_enc)
    return model.predict_proba(X_test_scaled)[:, 1]


def train_decision_tree(X_train, y_train, params):
    """Decision Tree 모델 학습"""
    X_train_enc, _ = encode_categorical_features(X_train, X_train)
    
    model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        random_state=42
    )
    model.fit(X_train_enc, y_train)
    
    return model


def predict_decision_tree(model, X_test):
    """Decision Tree 예측"""
    X_test_enc, _ = encode_categorical_features(X_test, X_test)
    return model.predict_proba(X_test_enc)[:, 1]


def train_random_forest(X_train, y_train, params):
    """Random Forest 모델 학습"""
    X_train_enc, _ = encode_categorical_features(X_train, X_train)
    
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_enc, y_train)
    
    return model


def predict_random_forest(model, X_test):
    """Random Forest 예측"""
    X_test_enc, _ = encode_categorical_features(X_test, X_test)
    return model.predict_proba(X_test_enc)[:, 1]


def train_xgboost(X_train, y_train, params):
    """XGBoost 모델 학습"""
    X_train_enc, _ = encode_categorical_features(X_train, X_train)
    
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params.get("learning_rate", 0.1),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train_enc, y_train)
    
    return model


def predict_xgboost(model, X_test):
    """XGBoost 예측"""
    X_test_enc, _ = encode_categorical_features(X_test, X_test)
    return model.predict_proba(X_test_enc)[:, 1]


def train_and_predict_ensemble(best_models, X_train_imp, y_train, X_test_imp):
    """모델 학습 및 예측"""
    models = {}
    scalers = {}
    predictions = {}
    weights = {}
    
    for model_type, info in best_models.items():
        params = info["params"]
        
        if model_type == "LogisticRegression":
            model, scaler = train_logistic_regression(X_train_imp, y_train, params)
            models[model_type] = model
            scalers[model_type] = scaler
            
        elif model_type == "DecisionTree":
            model = train_decision_tree(X_train_imp, y_train, params)
            models[model_type] = model
            
        elif model_type == "RandomForest":
            model = train_random_forest(X_train_imp, y_train, params)
            models[model_type] = model
            
        elif model_type == "XGBoost":
            model = train_xgboost(X_train_imp, y_train, params)
            models[model_type] = model
    
    # 예측 및 가중치 계산
    for model_type, model in models.items():
        if model_type == "LogisticRegression":
            y_prob = predict_logistic_regression(model, scalers[model_type], X_test_imp)
        elif model_type == "DecisionTree":
            y_prob = predict_decision_tree(model, X_test_imp)
        elif model_type == "RandomForest":
            y_prob = predict_random_forest(model, X_test_imp)
        elif model_type == "XGBoost":
            y_prob = predict_xgboost(model, X_test_imp)
        
        predictions[model_type] = y_prob
        base_score = best_models[model_type]["test_score"]
        weights[model_type] = base_score ** 2
    
    # 가중치 정규화
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # 가중합 앙상블
    ensemble_prob = np.zeros(len(X_test_imp))
    for model_type, y_prob in predictions.items():
        ensemble_prob += normalized_weights[model_type] * y_prob
    
    return ensemble_prob, normalized_weights, predictions


def main():
    """메인 실행 함수"""
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    results_dir = base_dir / "results"
    results_path = results_dir / "overfitting_experiments.json"
    
    print("=" * 70)
    print("Decision Tree 포함 여부에 따른 앙상블 성능 비교 실험")
    print("=" * 70)
    
    # 데이터 로드 및 전처리
    print("\n1. 데이터 로드 및 전처리 중...")
    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    categorical_cols = infer_categorical_from_dtype(X_train)
    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)
    print(f"   Train: {X_train_imp.shape[0]} samples, {X_test_imp.shape[1]} features")
    print(f"   Test: {X_test_imp.shape[0]} samples")
    
    # 실험 1: Decision Tree 포함
    print("\n" + "=" * 70)
    print("실험 1: Decision Tree 포함 앙상블")
    print("=" * 70)
    
    best_models_with_dt = find_best_models(results_path, exclude_dt=False)
    print("\n선정된 모델:")
    for model_type, info in best_models_with_dt.items():
        print(f"  {model_type}: {info['config_name']} (Score: {info['test_score']:.4f})")
    
    print("\n모델 학습 및 예측 중...")
    ensemble_prob_with_dt, weights_with_dt, predictions_with_dt = train_and_predict_ensemble(
        best_models_with_dt, X_train_imp, y_train, X_test_imp
    )
    
    metrics_with_dt = evaluate_model(y_test, ensemble_prob_with_dt)
    
    print("\n가중치:")
    for model_type, weight in weights_with_dt.items():
        print(f"  {model_type}: {weight:.4f}")
    
    print(f"\n앙상블 성능:")
    print(f"  Test Score: {metrics_with_dt['score']:.4f}")
    print(f"  F1-score: {metrics_with_dt['f1']:.4f}")
    print(f"  AUROC: {metrics_with_dt['auroc']:.4f}")
    
    # 실험 2: Decision Tree 제외
    print("\n" + "=" * 70)
    print("실험 2: Decision Tree 제외 앙상블")
    print("=" * 70)
    
    best_models_without_dt = find_best_models(results_path, exclude_dt=True)
    print("\n선정된 모델:")
    for model_type, info in best_models_without_dt.items():
        print(f"  {model_type}: {info['config_name']} (Score: {info['test_score']:.4f})")
    
    print("\n모델 학습 및 예측 중...")
    ensemble_prob_without_dt, weights_without_dt, predictions_without_dt = train_and_predict_ensemble(
        best_models_without_dt, X_train_imp, y_train, X_test_imp
    )
    
    metrics_without_dt = evaluate_model(y_test, ensemble_prob_without_dt)
    
    print("\n가중치:")
    for model_type, weight in weights_without_dt.items():
        print(f"  {model_type}: {weight:.4f}")
    
    print(f"\n앙상블 성능:")
    print(f"  Test Score: {metrics_without_dt['score']:.4f}")
    print(f"  F1-score: {metrics_without_dt['f1']:.4f}")
    print(f"  AUROC: {metrics_without_dt['auroc']:.4f}")
    
    # 비교 결과
    print("\n" + "=" * 70)
    print("비교 결과")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        "Decision Tree 포함": [
            metrics_with_dt['score'],
            metrics_with_dt['f1'],
            metrics_with_dt['auroc'],
            len(best_models_with_dt)
        ],
        "Decision Tree 제외": [
            metrics_without_dt['score'],
            metrics_without_dt['f1'],
            metrics_without_dt['auroc'],
            len(best_models_without_dt)
        ],
        "차이": [
            metrics_with_dt['score'] - metrics_without_dt['score'],
            metrics_with_dt['f1'] - metrics_without_dt['f1'],
            metrics_with_dt['auroc'] - metrics_without_dt['auroc'],
            len(best_models_with_dt) - len(best_models_without_dt)
        ]
    }, index=["Test Score", "F1-score", "AUROC", "모델 개수"])
    
    print("\n" + comparison.to_string())
    
    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)
    
    if metrics_with_dt['score'] > metrics_without_dt['score']:
        print("✓ Decision Tree를 포함한 앙상블이 더 좋은 성능을 보입니다.")
        print(f"  성능 향상: {metrics_with_dt['score'] - metrics_without_dt['score']:.4f}")
        recommendation = "Decision Tree 포함 권장"
    else:
        print("✗ Decision Tree를 제외한 앙상블이 더 좋은 성능을 보입니다.")
        print(f"  성능 향상: {metrics_without_dt['score'] - metrics_with_dt['score']:.4f}")
        recommendation = "Decision Tree 제외 권장"
    
    print(f"\n권장사항: {recommendation}")
    
    # 결과 저장
    comparison_result = {
        "with_decision_tree": {
            "models": list(best_models_with_dt.keys()),
            "weights": weights_with_dt,
            "metrics": metrics_with_dt
        },
        "without_decision_tree": {
            "models": list(best_models_without_dt.keys()),
            "weights": weights_without_dt,
            "metrics": metrics_without_dt
        },
        "comparison": {
            "score_diff": float(metrics_with_dt['score'] - metrics_without_dt['score']),
            "f1_diff": float(metrics_with_dt['f1'] - metrics_without_dt['f1']),
            "auroc_diff": float(metrics_with_dt['auroc'] - metrics_without_dt['auroc']),
            "recommendation": recommendation
        }
    }
    
    output_path = results_dir / "ensemble_comparison_dt.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")
    
    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

