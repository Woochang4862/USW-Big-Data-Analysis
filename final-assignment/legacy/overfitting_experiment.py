#!/usr/bin/env python
"""
모델 과적합 실험 파이프라인
- raw 데이터를 사용하여 간단한 모델부터 복잡한 모델까지 점진적으로 과적합 실험
- Train/Test 성능 차이를 통해 과적합 정도 측정
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural Network experiments will be skipped.")


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


def evaluate_model(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
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


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """범주형 변수를 원-핫 인코딩"""
    categorical_cols = infer_categorical_from_dtype(X_train)
    
    # 범주형 변수가 있으면 원-핫 인코딩
    if categorical_cols:
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=False)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=False)
        
        # Train에만 있는 컬럼 제거 (Test에 맞춤)
        common_cols = X_train_encoded.columns.intersection(X_test_encoded.columns)
        X_train_encoded = X_train_encoded[common_cols]
        X_test_encoded = X_test_encoded[common_cols]
        
        return X_train_encoded, X_test_encoded
    
    return X_train, X_test


def experiment_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict]:
    """Logistic Regression 실험 (3단계)"""
    results = []
    
    # 범주형 변수 인코딩
    X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test)
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    # 25개 범주: C 값을 더 단순한 쪽으로 확장 (100000부터 0.0001까지)
    configs = [
        {"name": "Level-10", "C": 100000.0},  # 매우 약한 정규화 (가장 단순)
        {"name": "Level-9", "C": 50000.0},
        {"name": "Level-8", "C": 20000.0},
        {"name": "Level-7", "C": 10000.0},
        {"name": "Level-6", "C": 5000.0},
        {"name": "Level-5", "C": 2000.0},
        {"name": "Level-4", "C": 1000.0},
        {"name": "Level-3", "C": 500.0},
        {"name": "Level-2", "C": 200.0},
        {"name": "Level-1", "C": 100.0},
        {"name": "Level0", "C": 50.0},
        {"name": "Level1", "C": 20.0},
        {"name": "Level2", "C": 10.0},
        {"name": "Level3", "C": 5.0},
        {"name": "Level4", "C": 1.0},
        {"name": "Level5", "C": 0.5},
        {"name": "Level6", "C": 0.1},
        {"name": "Level7", "C": 0.05},
        {"name": "Level8", "C": 0.01},
        {"name": "Level9", "C": 0.005},
        {"name": "Level10", "C": 0.001},
        {"name": "Level11", "C": 0.0005},
        {"name": "Level12", "C": 0.0001},
        {"name": "Level13", "C": 0.00005},
        {"name": "Level14", "C": 0.00001},  # 매우 강한 정규화 (가장 복잡)
    ]
    
    for config in configs:
        model = LogisticRegression(C=config["C"], max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        train_prob = model.predict_proba(X_train_scaled)[:, 1]
        test_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        train_metrics = evaluate_model(y_train, train_prob)
        test_metrics = evaluate_model(y_test, test_prob)
        result = build_result(
            model_type="LogisticRegression",
            config_name=config["name"],
            params={"C": config["C"]},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        results.append(result)
        print(
            f"  {config['name']}: "
            f"Train F1={train_metrics['f1']:.4f} / AUROC={train_metrics['auroc']:.4f} / Score={result['train_score']:.4f} | "
            f"Test F1={test_metrics['f1']:.4f} / AUROC={test_metrics['auroc']:.4f} / Score={result['test_score']:.4f} | "
            f"Score Gap={result['overfitting_gap_score']:.4f}"
        )
    
    return results


def experiment_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict]:
    """Decision Tree 실험 (4단계)"""
    results = []
    
    # 범주형 변수 인코딩
    X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test)
    
    # 5개 범주를 Level 형식으로 변경
    configs = [
        {"name": "Level1", "max_depth": 1},  # 매우 단순
        {"name": "Level2", "max_depth": 5},
        {"name": "Level3", "max_depth": 15},
        {"name": "Level4", "max_depth": 50},  # 더 깊게
        {"name": "Level5", "max_depth": None},  # 무제한
    ]
    
    for config in configs:
        model = DecisionTreeClassifier(max_depth=config["max_depth"], random_state=42)
        model.fit(X_train_enc, y_train)
        
        train_prob = model.predict_proba(X_train_enc)[:, 1]
        test_prob = model.predict_proba(X_test_enc)[:, 1]
        
        train_metrics = evaluate_model(y_train, train_prob)
        test_metrics = evaluate_model(y_test, test_prob)
        result = build_result(
            model_type="DecisionTree",
            config_name=config["name"],
            params={"max_depth": config["max_depth"]},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        results.append(result)
        print(
            f"  {config['name']}: "
            f"Train Score={result['train_score']:.4f} | "
            f"Test Score={result['test_score']:.4f} | "
            f"Score Gap={result['overfitting_gap_score']:.4f}"
        )
    
    return results


def experiment_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict]:
    """Random Forest 실험 (4단계)"""
    results = []
    
    # 범주형 변수 인코딩
    X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test)
    
    # 25개 범주: 더 단순한 쪽으로 확장
    configs = [
        {"name": "Level-10", "n_estimators": 1, "max_depth": 1},   # 매우 단순
        {"name": "Level-9", "n_estimators": 2, "max_depth": 1},
        {"name": "Level-8", "n_estimators": 3, "max_depth": 1},
        {"name": "Level-7", "n_estimators": 4, "max_depth": 1},
        {"name": "Level-6", "n_estimators": 5, "max_depth": 2},
        {"name": "Level-5", "n_estimators": 7, "max_depth": 2},
        {"name": "Level-4", "n_estimators": 10, "max_depth": 2},
        {"name": "Level-3", "n_estimators": 15, "max_depth": 3},
        {"name": "Level-2", "n_estimators": 20, "max_depth": 3},
        {"name": "Level-1", "n_estimators": 30, "max_depth": 4},
        {"name": "Level0", "n_estimators": 50, "max_depth": 5},
        {"name": "Level1", "n_estimators": 70, "max_depth": 7},
        {"name": "Level2", "n_estimators": 100, "max_depth": 10},
        {"name": "Level3", "n_estimators": 150, "max_depth": 12},
        {"name": "Level4", "n_estimators": 200, "max_depth": 15},
        {"name": "Level5", "n_estimators": 300, "max_depth": 20},
        {"name": "Level6", "n_estimators": 500, "max_depth": 25},
        {"name": "Level7", "n_estimators": 700, "max_depth": 30},
        {"name": "Level8", "n_estimators": 1000, "max_depth": 40},
        {"name": "Level9", "n_estimators": 1500, "max_depth": 50},
        {"name": "Level10", "n_estimators": 2000, "max_depth": None},  # 매우 복잡
        {"name": "Level11", "n_estimators": 2500, "max_depth": None},
        {"name": "Level12", "n_estimators": 3000, "max_depth": None},
        {"name": "Level13", "n_estimators": 4000, "max_depth": None},
        {"name": "Level14", "n_estimators": 5000, "max_depth": None},
    ]
    
    for config in configs:
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_enc, y_train)
        
        train_prob = model.predict_proba(X_train_enc)[:, 1]
        test_prob = model.predict_proba(X_test_enc)[:, 1]
        
        train_metrics = evaluate_model(y_train, train_prob)
        test_metrics = evaluate_model(y_test, test_prob)
        result = build_result(
            model_type="RandomForest",
            config_name=config["name"],
            params={"n_estimators": config["n_estimators"], "max_depth": config["max_depth"]},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        results.append(result)
        print(
            f"  {config['name']}: "
            f"Train Score={result['train_score']:.4f} | "
            f"Test Score={result['test_score']:.4f} | "
            f"Score Gap={result['overfitting_gap_score']:.4f}"
        )
    
    return results


def experiment_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict]:
    """XGBoost 실험 (4단계)"""
    results = []
    
    # 범주형 변수 인코딩
    X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test)
    
    # 25개 범주: 더 단순한 쪽으로 확장
    configs = [
        {"name": "Level-10", "n_estimators": 1, "max_depth": 1, "learning_rate": 0.1},   # 매우 단순
        {"name": "Level-9", "n_estimators": 2, "max_depth": 1, "learning_rate": 0.1},
        {"name": "Level-8", "n_estimators": 3, "max_depth": 1, "learning_rate": 0.1},
        {"name": "Level-7", "n_estimators": 5, "max_depth": 1, "learning_rate": 0.1},
        {"name": "Level-6", "n_estimators": 7, "max_depth": 2, "learning_rate": 0.1},
        {"name": "Level-5", "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1},
        {"name": "Level-4", "n_estimators": 15, "max_depth": 2, "learning_rate": 0.1},
        {"name": "Level-3", "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
        {"name": "Level-2", "n_estimators": 30, "max_depth": 3, "learning_rate": 0.1},
        {"name": "Level-1", "n_estimators": 40, "max_depth": 4, "learning_rate": 0.1},
        {"name": "Level0", "n_estimators": 50, "max_depth": 5, "learning_rate": 0.1},
        {"name": "Level1", "n_estimators": 80, "max_depth": 6, "learning_rate": 0.1},
        {"name": "Level2", "n_estimators": 100, "max_depth": 7, "learning_rate": 0.1},
        {"name": "Level3", "n_estimators": 150, "max_depth": 9, "learning_rate": 0.1},
        {"name": "Level4", "n_estimators": 200, "max_depth": 11, "learning_rate": 0.1},
        {"name": "Level5", "n_estimators": 300, "max_depth": 14, "learning_rate": 0.1},
        {"name": "Level6", "n_estimators": 500, "max_depth": 18, "learning_rate": 0.1},
        {"name": "Level7", "n_estimators": 700, "max_depth": 22, "learning_rate": 0.1},
        {"name": "Level8", "n_estimators": 1000, "max_depth": 25, "learning_rate": 0.1},
        {"name": "Level9", "n_estimators": 1500, "max_depth": 28, "learning_rate": 0.2},
        {"name": "Level10", "n_estimators": 2000, "max_depth": 30, "learning_rate": 0.5},  # 매우 복잡
        {"name": "Level11", "n_estimators": 2500, "max_depth": 30, "learning_rate": 0.5},
        {"name": "Level12", "n_estimators": 3000, "max_depth": 30, "learning_rate": 0.5},
        {"name": "Level13", "n_estimators": 4000, "max_depth": 30, "learning_rate": 0.5},
        {"name": "Level14", "n_estimators": 5000, "max_depth": 30, "learning_rate": 0.5},
    ]
    
    for config in configs:
        model = xgb.XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        model.fit(X_train_enc, y_train)
        
        train_prob = model.predict_proba(X_train_enc)[:, 1]
        test_prob = model.predict_proba(X_test_enc)[:, 1]
        
        train_metrics = evaluate_model(y_train, train_prob)
        test_metrics = evaluate_model(y_test, test_prob)
        result = build_result(
            model_type="XGBoost",
            config_name=config["name"],
            params={
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"],
            },
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        results.append(result)
        print(
            f"  {config['name']}: "
            f"Train Score={result['train_score']:.4f} | "
            f"Test Score={result['test_score']:.4f} | "
            f"Score Gap={result['overfitting_gap_score']:.4f}"
        )
    
    return results


def experiment_neural_network(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict]:
    """Neural Network 실험 (4단계)"""
    results = []
    
    if not TF_AVAILABLE:
        print("  Skipping Neural Network experiments (TensorFlow not available)")
        return results
    
    # 범주형 변수 인코딩
    X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test)
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    # 5개 범주를 Level 형식으로 변경
    configs = [
        {"name": "Level1", "layers": [16]},  # 매우 단순
        {"name": "Level2", "layers": [64]},
        {"name": "Level3", "layers": [128, 64]},
        {"name": "Level4", "layers": [512, 256, 128, 64]},  # 더 복잡하게
        {"name": "Level5", "layers": [1024, 512, 256, 128, 64, 32]},  # 매우 복잡 (6층)
    ]
    
    for config in configs:
        # 모델 생성
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
        
        for units in config["layers"]:
            model.add(layers.Dense(units, activation="relu"))
        
        model.add(layers.Dense(1, activation="sigmoid"))
        
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # 학습
        epochs = {
            "Level1": 50,
            "Level2": 50,
            "Level3": 100,
            "Level4": 200,
            "Level5": 300
        }[config["name"]]
        model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=64,
            verbose=0,
            validation_split=0.1
        )
        
        train_prob = model.predict(X_train_scaled, verbose=0).flatten()
        test_prob = model.predict(X_test_scaled, verbose=0).flatten()
        
        train_metrics = evaluate_model(y_train, train_prob)
        test_metrics = evaluate_model(y_test, test_prob)
        result = build_result(
            model_type="NeuralNetwork",
            config_name=config["name"],
            params={"layers": config["layers"], "epochs": epochs},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        results.append(result)
        print(
            f"  {config['name']}: "
            f"Train Score={result['train_score']:.4f} | "
            f"Test Score={result['test_score']:.4f} | "
            f"Score Gap={result['overfitting_gap_score']:.4f}"
        )
    
    return results


def load_existing_results(results_dir: Path) -> List[Dict]:
    """기존 결과 파일 로드"""
    json_path = results_dir / "overfitting_experiments.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(all_results: List[Dict], results_dir: Path, update_models: List[str] = None):
    """결과를 JSON과 CSV로 저장
    
    Args:
        all_results: 새로 실험한 결과
        results_dir: 결과 저장 디렉토리
        update_models: 업데이트할 모델 리스트 (None이면 전체 교체)
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 결과 로드 (선택적 업데이트인 경우)
    if update_models is not None:
        existing_results = load_existing_results(results_dir)
        # 기존 결과에서 업데이트할 모델 제거
        existing_results = [
            r for r in existing_results 
            if r.get("model_type") not in update_models
        ]
        # 새 결과와 병합
        all_results = existing_results + all_results
        print(f"\n기존 결과에서 {update_models} 모델만 업데이트합니다.")
    
    # JSON 저장
    json_path = results_dir / "overfitting_experiments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {json_path}")
    
    # CSV 요약 테이블 생성
    summary_data = []
    for result in all_results:
        summary_data.append({
            "model_type": result["model_type"],
            "config_name": result["config_name"],
            "train_f1": result["train_f1"],
            "test_f1": result["test_f1"],
            "overfitting_gap_f1": result["overfitting_gap_f1"],
            "train_auroc": result["train_auroc"],
            "test_auroc": result["test_auroc"],
            "overfitting_gap_auroc": result["overfitting_gap_auroc"],
            "train_score": result["train_score"],
            "test_score": result["test_score"],
            "overfitting_gap_score": result["overfitting_gap_score"],
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = results_dir / "overfitting_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"요약 저장: {csv_path}")
    
    # 요약 출력
    print("\n=== 실험 요약 ===")
    print(summary_df.to_string(index=False))


def main():
    """메인 실행 함수"""
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    results_dir = base_dir / "results"
    
    print("=" * 60)
    print("모델 과적합 실험 시작")
    print("=" * 60)
    
    # 데이터 로드
    print("\n1. 데이터 로드 중...")
    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # 전처리
    print("\n2. 데이터 전처리 중...")
    categorical_cols = infer_categorical_from_dtype(X_train)
    print(f"   범주형 변수: {len(categorical_cols)}개")
    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)
    print("   결측치 처리 완료")
    
    # 실험 실행 (선택적 모델만)
    all_results = []
    update_models = ["LogisticRegression", "RandomForest", "XGBoost"]  # 업데이트할 모델
    
    print("\n3. Logistic Regression 실험 중...")
    all_results.extend(experiment_logistic_regression(X_train_imp, y_train, X_test_imp, y_test))
    
    print("\n4. Decision Tree 실험 중... (건너뜀 - 기존 결과 유지)")
    
    print("\n5. Random Forest 실험 중...")
    all_results.extend(experiment_random_forest(X_train_imp, y_train, X_test_imp, y_test))
    
    print("\n6. XGBoost 실험 중...")
    all_results.extend(experiment_xgboost(X_train_imp, y_train, X_test_imp, y_test))
    
    print("\n7. Neural Network 실험 중... (건너뜀 - 시간 절약)")
    
    # 결과 저장 (선택적 업데이트)
    print("\n8. 결과 저장 중...")
    save_results(all_results, results_dir, update_models=update_models)
    
    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

