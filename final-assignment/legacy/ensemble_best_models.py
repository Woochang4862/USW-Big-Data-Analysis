#!/usr/bin/env python
"""
최고 성능 모델 앙상블
- overfitting_experiments.json에서 각 모델 타입별 최고 test_score를 가진 하이퍼파라미터 선정
- 해당 하이퍼파라미터로 모델 재학습 및 앙상블 예측
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural Network will be skipped.")


# overfitting_experiment.py의 함수들 재사용
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
    
    # 범주형 변수: 최빈값으로 채우기
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
    
    # 수치형 변수: 중앙값으로 채우기
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


def find_best_models(results_path: Path, exclude_models: List[str] = None) -> Dict[str, Dict]:
    """각 모델 타입별 최고 test_score를 가진 설정 찾기"""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 제외할 모델 타입
    if exclude_models is None:
        exclude_models = ["NeuralNetwork", "DecisionTree"]  # 기본값: DecisionTree 제외
    
    best_models = {}
    
    for result in results:
        model_type = result["model_type"]
        
        if model_type in exclude_models:
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


def train_neural_network(X_train, y_train, params):
    """Neural Network 모델 학습"""
    if not TF_AVAILABLE:
        return None, None
    
    X_train_enc, _ = encode_categorical_features(X_train, X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
    
    for units in params["layers"]:
        model.add(layers.Dense(units, activation="relu"))
    
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    epochs = params.get("epochs", 50)
    model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=64,
        verbose=0,
        validation_split=0.1
    )
    
    return model, scaler


def predict_neural_network(model, scaler, X_test):
    """Neural Network 예측"""
    if model is None:
        return None
    
    X_test_enc, _ = encode_categorical_features(X_test, X_test)
    X_test_scaled = scaler.transform(X_test_enc)
    return model.predict(X_test_scaled, verbose=0).flatten()


def stacking_ensemble(best_models, X_train_imp, y_train, X_test_imp, n_folds=5):
    """스태킹 앙상블 구현
    
    Args:
        best_models: 최고 성능 모델 딕셔너리
        X_train_imp: 학습 데이터
        y_train: 학습 타겟
        X_test_imp: 테스트 데이터
        n_folds: Cross-validation 폴드 수
    
    Returns:
        ensemble_prob: 앙상블 예측 확률
        meta_model: 메타 모델
        oof_predictions: OOF 예측 (각 모델별)
        test_predictions: 테스트 예측 (각 모델별)
    """
    print(f"\n스태킹 앙상블 (CV folds: {n_folds})")
    
    # Cross-validation 설정
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # OOF 예측 저장소
    oof_predictions = {model_type: np.zeros(len(X_train_imp)) for model_type in best_models.keys()}
    test_predictions = {model_type: [] for model_type in best_models.keys()}
    
    # Base models 학습 및 OOF 예측 생성
    print("\n  Base models 학습 및 OOF 예측 생성 중...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_imp, y_train), 1):
        print(f"    Fold {fold}/{n_folds}...")
        
        X_tr, X_val = X_train_imp.iloc[train_idx], X_train_imp.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        fold_test_preds = {}
        
        for model_type, info in best_models.items():
            params = info["params"]
            
            # 모델 학습
            if model_type == "LogisticRegression":
                model, scaler = train_logistic_regression(X_tr, y_tr, params)
                # Validation 예측
                oof_pred = predict_logistic_regression(model, scaler, X_val)
                # Test 예측 (나중에 평균)
                test_pred = predict_logistic_regression(model, scaler, X_test_imp)
                
            elif model_type == "RandomForest":
                model = train_random_forest(X_tr, y_tr, params)
                oof_pred = predict_random_forest(model, X_val)
                test_pred = predict_random_forest(model, X_test_imp)
                
            elif model_type == "XGBoost":
                model = train_xgboost(X_tr, y_tr, params)
                oof_pred = predict_xgboost(model, X_val)
                test_pred = predict_xgboost(model, X_test_imp)
            
            # OOF 예측 저장
            oof_predictions[model_type][val_idx] = oof_pred
            fold_test_preds[model_type] = test_pred
        
        # Fold별 테스트 예측 저장
        for model_type, pred in fold_test_preds.items():
            test_predictions[model_type].append(pred)
    
    # 테스트 예측 평균 (각 모델별)
    for model_type in test_predictions:
        test_predictions[model_type] = np.mean(test_predictions[model_type], axis=0)
    
    # 메타 피처 생성 (OOF 예측들을 결합)
    print("\n  메타 모델 학습 중...")
    meta_X_train = np.column_stack([oof_predictions[mt] for mt in best_models.keys()])
    meta_X_test = np.column_stack([test_predictions[mt] for mt in best_models.keys()])
    
    # 메타 모델 (Logistic Regression)
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X_train, y_train)
    
    # 최종 예측
    ensemble_prob = meta_model.predict_proba(meta_X_test)[:, 1]
    
    return ensemble_prob, meta_model, oof_predictions, test_predictions


def main(case: str = None, exclude_models: List[str] = None):
    """메인 실행 함수
    
    Args:
        case: 케이스 이름 (결과 파일명에 사용)
        exclude_models: 제외할 모델 리스트
    """
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    results_dir = base_dir / "results"
    results_path = results_dir / "overfitting_experiments.json"
    
    if case:
        print("=" * 60)
        print(f"앙상블 케이스: {case}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("최고 성능 모델 앙상블")
        print("=" * 60)
    
    # 최고 모델 찾기
    print("\n1. 최고 성능 모델 선정 중...")
    best_models = find_best_models(results_path, exclude_models=exclude_models)
    
    for model_type, info in best_models.items():
        print(f"\n{model_type}:")
        print(f"  Config: {info['config_name']}")
        print(f"  Params: {info['params']}")
        print(f"  Test Score: {info['test_score']:.4f} (F1: {info['test_f1']:.4f}, AUROC: {info['test_auroc']:.4f})")
    
    # 데이터 로드 및 전처리
    print("\n2. 데이터 로드 및 전처리 중...")
    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    categorical_cols = infer_categorical_from_dtype(X_train)
    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)
    print(f"   Train: {X_train_imp.shape[0]} samples, {X_test_imp.shape[1]} features")
    print(f"   Test: {X_test_imp.shape[0]} samples")
    
    # 모델 학습
    print("\n3. 최고 성능 모델 학습 중...")
    models = {}
    scalers = {}
    
    for model_type, info in best_models.items():
        params = info["params"]
        print(f"\n  {model_type} 학습 중...")
        
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
    
    # 개별 모델 평가
    print("\n4. 개별 모델 평가 중...")
    individual_results = {}
    
    for model_type, model in models.items():
        if model_type == "LogisticRegression":
            y_prob = predict_logistic_regression(model, scalers[model_type], X_test_imp)
        elif model_type == "DecisionTree":
            y_prob = predict_decision_tree(model, X_test_imp)
        elif model_type == "RandomForest":
            y_prob = predict_random_forest(model, X_test_imp)
        elif model_type == "XGBoost":
            y_prob = predict_xgboost(model, X_test_imp)
        
        metrics = evaluate_model(y_test, y_prob)
        individual_results[model_type] = metrics
        print(f"  {model_type}: F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}, Score={metrics['score']:.4f}")
    
    # 앙상블 방법 선택
    use_stacking = case and "stacking" in case.lower()
    
    if use_stacking and len(best_models) >= 2:
        # 스태킹 앙상블
        print("\n5. 스태킹 앙상블 예측 중...")
        ensemble_prob, meta_model, oof_predictions, test_predictions = stacking_ensemble(
            best_models, X_train_imp, y_train, X_test_imp, n_folds=5
        )
        
        # OOF 예측으로 메타 모델 평가
        meta_X_train = np.column_stack([oof_predictions[mt] for mt in best_models.keys()])
        oof_meta_pred = meta_model.predict_proba(meta_X_train)[:, 1]
        oof_metrics = evaluate_model(y_train, oof_meta_pred)
        
        print(f"\n  OOF 메타 모델 성능:")
        print(f"    Train Score: {oof_metrics['score']:.4f}")
        print(f"    F1-score: {oof_metrics['f1']:.4f}")
        print(f"    AUROC: {oof_metrics['auroc']:.4f}")
        
        ensemble_metrics = evaluate_model(y_test, ensemble_prob)
        normalized_weights = {}  # 스태킹은 가중치 없음
        
    else:
        # 가중합 앙상블
        print("\n5. 앙상블 예측 중... (가중합)")
        predictions = {}
        weights = {}
        
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
            # 각 모델의 test_score를 제곱하여 가중치로 사용 (성능 차이를 더 크게 반영)
            base_score = best_models[model_type]["test_score"]
            # test_score를 제곱하여 성능이 좋은 모델에 더 높은 가중치 부여
            weights[model_type] = base_score ** 2
        
        # 가중치 정규화 (합이 1이 되도록)
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        print("\n  가중치 (test_score^2 기반):")
        for model_type, weight in normalized_weights.items():
            base_score = best_models[model_type]["test_score"]
            print(f"    {model_type}: {weight:.4f} (원래 Score: {base_score:.4f}, Score^2: {base_score**2:.4f})")
        
        # 가중합 앙상블
        ensemble_prob = np.zeros(len(X_test_imp))
        for model_type, y_prob in predictions.items():
            ensemble_prob += normalized_weights[model_type] * y_prob
        
        ensemble_metrics = evaluate_model(y_test, ensemble_prob)
        meta_model = None
        oof_predictions = {}
        test_predictions = {}
    
    print("\n" + "=" * 60)
    print("앙상블 결과")
    print("=" * 60)
    print(f"앙상블 Test Score: {ensemble_metrics['score']:.4f}")
    print(f"  F1-score: {ensemble_metrics['f1']:.4f}")
    print(f"  AUROC: {ensemble_metrics['auroc']:.4f}")
    
    # Submission 파일 생성
    print("\n6. Submission 파일 생성 중...")
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        "ID": X_test_imp.index,
        "HE_D3_label": ensemble_prob
    })
    submission_df = submission_df.set_index("ID")
    
    submission_path = base_dir / f"submission_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=True)
    print(f"Submission 파일 저장: {submission_path}")
    
    # 타임스탬프 없는 기본 파일도 저장 (기존 호환성)
    default_path = base_dir / "submission.csv"
    submission_df.to_csv(default_path, index=True)
    print(f"기본 Submission 파일 저장: {default_path}")
    
    # 결과 저장
    ensemble_result = {
        "best_models": best_models,
        "individual_results": individual_results,
        "ensemble_result": ensemble_metrics,
        "ensemble_method": "stacking" if use_stacking else "weighted_average",
        "weights": normalized_weights if not use_stacking else {}
    }
    
    if use_stacking:
        ensemble_result["meta_model"] = {
            "type": "LogisticRegression",
            "C": 1.0
        }
    
    # 결과 파일명 결정
    if case:
        output_path = results_dir / f"ensemble_results_{case}.json"
    else:
        output_path = results_dir / "ensemble_results.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_result, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_path}")
    
    # 개별 모델과 비교
    print("\n" + "=" * 60)
    print("개별 모델 vs 앙상블 비교")
    print("=" * 60)
    print(f"{'Model':<20} {'F1':<8} {'AUROC':<8} {'Score':<8}")
    print("-" * 60)
    for model_type, metrics in individual_results.items():
        print(f"{model_type:<20} {metrics['f1']:<8.4f} {metrics['auroc']:<8.4f} {metrics['score']:<8.4f}")
    print("-" * 60)
    print(f"{'Ensemble (Average)':<20} {ensemble_metrics['f1']:<8.4f} {ensemble_metrics['auroc']:<8.4f} {ensemble_metrics['score']:<8.4f}")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, help="케이스 이름")
    parser.add_argument("--exclude", nargs="+", help="제외할 모델 리스트")
    args = parser.parse_args()
    
    exclude_models = None
    if args.exclude:
        exclude_models = args.exclude
    
    main(case=args.case, exclude_models=exclude_models)

