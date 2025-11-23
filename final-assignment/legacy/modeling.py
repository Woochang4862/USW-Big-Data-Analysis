#!/usr/bin/env python
"""
RandomForest 기반 HE_D3_label 분류 파이프라인.
- final-assignment/*.csv를 통합한 combined_dataset.csv 생성
- 주어진 결측치 처리 규칙 적용 (범주형: 최빈값, 수치형: 중앙값)
- 기본 학습(Train 전용) vs 전체 학습(Train+Test) 성능 비교

사용 예시:
    python modeling.py \
        --base-dir /Users/jeong-uchang/USW-Big-Data-Analysis/final-assignment

옵션:
    --meta-path : updated_dataframe.xlsx 위치 (없으면 dtype 기반 자동 추론)
    --n-estimators, --max-depth 등 RandomForest 하이퍼파라미터
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="final-assignment 모델 학습 스크립트")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="데이터가 위치한 디렉터리",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="updated_dataframe.xlsx 경로 (미지정 시 base-dir에서 자동 탐색)",
    )
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_data_dirs(base_dir: Path) -> Tuple[Path, Path]:
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def load_feature_label_pairs(raw_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(raw_dir / "X_train.csv", index_col="ID")
    y_train = pd.read_csv(raw_dir / "Y_train.csv", index_col="ID")["HE_D3_label"]
    X_test = pd.read_csv(raw_dir / "X_test.csv", index_col="ID")
    y_test = pd.read_csv(raw_dir / "Y_test.csv", index_col="ID")["HE_D3_label"]
    return X_train, y_train, X_test, y_test


def ensure_combined_csv(
    processed_dir: Path, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Path:
    combined_path = processed_dir / "combined_dataset.csv"
    if combined_path.exists():
        return combined_path

    train = X_train.join(y_train.rename("HE_D3_label")).copy()
    train["split"] = "train"
    test = X_test.join(y_test.rename("HE_D3_label")).copy()
    test["split"] = "test"
    combined = pd.concat([train, test], axis=0)
    combined.to_csv(combined_path, index=True)
    return combined_path


def read_meta_categories(meta_path: Path, feature_names: Iterable[str]) -> Iterable[str]:
    df_meta = pd.read_excel(meta_path)
    categorical_variables = df_meta[df_meta["data type"] == "category"]["variable"].tolist()
    return list(set(feature_names).intersection(categorical_variables))


def infer_categorical_from_dtype(df: pd.DataFrame) -> Iterable[str]:
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
    categorical_cols: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
    }


def train_and_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    params: Dict,
) -> Dict[str, float]:
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return evaluate_predictions(y_eval, y_prob)


def cross_val_report(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    n_splits: int,
    random_state: int,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = RandomForestClassifier(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_metric = evaluate_predictions(y.iloc[val_idx], y_prob)
        metrics.append(fold_metric)

    return {
        "f1_mean": float(np.mean([m["f1"] for m in metrics])),
        "f1_std": float(np.std([m["f1"] for m in metrics], ddof=1)),
        "auroc_mean": float(np.mean([m["auroc"] for m in metrics])),
        "auroc_std": float(np.std([m["auroc"] for m in metrics], ddof=1)),
    }


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    raw_dir, processed_dir = ensure_data_dirs(base_dir)

    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    combined_path = ensure_combined_csv(processed_dir, X_train, y_train, X_test, y_test)

    meta_path = args.meta_path or (base_dir / "updated_dataframe.xlsx")
    if meta_path.exists():
        categorical_cols = read_meta_categories(meta_path, X_train.columns)
        meta_source = str(meta_path)
    else:
        categorical_cols = infer_categorical_from_dtype(X_train)
        meta_source = "dtype_inference"

    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)

    X_train_imp.to_csv(processed_dir / "X_train_imputed.csv", index=True)
    X_test_imp.to_csv(processed_dir / "X_test_imputed.csv", index=True)

    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "n_jobs": -1,
    }

    baseline_metrics = train_and_eval(X_train_imp, y_train, X_test_imp, y_test, model_params)

    X_train_imp, y_train = X_train_imp.align(y_train, join="inner", axis=0)
    X_test_imp, y_test = X_test_imp.align(y_test, join="inner", axis=0)

    X_full = pd.concat([X_train_imp, X_test_imp], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    X_full.to_csv(processed_dir / "X_full_imputed.csv", index=True)
    y_full.to_csv(processed_dir / "Y_full.csv", index=True)

    cv_train_only = cross_val_report(X_train_imp, y_train, model_params, args.n_folds, args.random_state)
    cv_full = cross_val_report(X_full, y_full, model_params, args.n_folds, args.random_state)

    results = {
        "meta_source": meta_source,
        "combined_csv": str(combined_path),
        "baseline_test_metrics": baseline_metrics,
        "cv_train_only": cv_train_only,
        "cv_train_plus_test": cv_full,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


