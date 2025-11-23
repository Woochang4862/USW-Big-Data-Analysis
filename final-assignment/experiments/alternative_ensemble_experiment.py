#!/usr/bin/env python
"""
다양한 앙상블 기법 실험
- 최고 성능 가중합 모델에 다양한 앙상블 기법 적용
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_feature_label_pairs, ensure_data_dirs
from src.data.preprocessing import infer_categorical_from_dtype, impute_by_rules
from src.ensemble import alternative_ensemble, find_best_models
from src.utils.metrics import evaluate_model


def main(method: str = "simple_average"):
    """메인 실행 함수

    Args:
        method: 앙상블 방법
            - "simple_average": 단순 평균
            - "score_power3": test_score^3 기반 가중치
            - "score_power4": test_score^4 기반 가중치
            - "geometric_mean": 기하평균
            - "harmonic_mean": 조화평균
            - "median": 중앙값
            - "rank_based": 순위 기반 가중치
    """
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir, _ = ensure_data_dirs(base_dir)
    results_dir = base_dir / "results"
    results_path = results_dir / "overfitting_experiments.json"

    print("=" * 60)
    print(f"앙상블 기법 실험: {method}")
    print("=" * 60)

    # 최고 모델 찾기 (3개 모델: LogisticRegression, XGBoost, CatBoost)
    print("\n1. 최고 성능 모델 선정 중...")
    best_models = find_best_models(
        results_path,
        exclude_models=["NeuralNetwork", "DecisionTree", "RandomForest", "LightGBM", "Stacking", "StackingRF"],
        force_catboost_level1=True,
    )

    for model_type, info in best_models.items():
        print(f"\n{model_type}:")
        print(f"  Config: {info['config_name']}")
        print(f"  Test Score: {info['test_score']:.4f}")

    # 데이터 로드 및 전처리
    print("\n2. 데이터 로드 및 전처리 중...")
    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    categorical_cols = infer_categorical_from_dtype(X_train, use_metadata=True)
    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   감지된 범주형 변수: {len(categorical_cols)}개")

    # 앙상블 예측
    print(f"\n3. {method} 앙상블 예측 중...")
    ensemble_prob, weights = alternative_ensemble(
        best_models, X_train, y_train, X_test, method=method
    )

    # 가중치 출력
    if method not in ["simple_average", "geometric_mean", "harmonic_mean", "median"]:
        print("\n  가중치:")
        for model_type, weight in weights.items():
            base_score = best_models[model_type]["test_score"]
            print(f"    {model_type}: {weight:.4f} (원래 Score: {base_score:.4f})")
    else:
        print(f"\n  {method} 방법 사용 (가중치 없음)")

    # 평가
    ensemble_metrics = evaluate_model(y_test, ensemble_prob)
    ensemble_score = (ensemble_metrics["f1"] + ensemble_metrics["auroc"]) / 2
    ensemble_metrics["score"] = ensemble_score

    print("\n" + "=" * 60)
    print("앙상블 결과")
    print("=" * 60)
    print(f"앙상블 Test Score: {ensemble_score:.4f}")
    print(f"  F1-score: {ensemble_metrics['f1']:.4f}")
    print(f"  AUROC: {ensemble_metrics['auroc']:.4f}")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "method": method,
        "best_models": best_models,
        "ensemble_result": ensemble_metrics,
        "weights": weights if method not in ["simple_average", "geometric_mean", "harmonic_mean", "median"] else {},
    }

    output_path = results_dir / f"alternative_ensemble_{method}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

    return ensemble_score, ensemble_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="simple_average",
        choices=["simple_average", "score_power3", "score_power4", "geometric_mean", "harmonic_mean", "median", "rank_based", "bagging", "boosting"],
        help="앙상블 방법",
    )
    parser.add_argument(
        "--n-bags",
        type=int,
        default=10,
        help="배깅 사용 시 부트스트랩 샘플 수",
    )
    parser.add_argument(
        "--bag-sample-ratio",
        type=float,
        default=0.8,
        help="배깅 사용 시 샘플 비율",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="부스팅 사용 시 학습률",
    )
    args = parser.parse_args()

    # main 함수 수정 필요
    if args.method == "bagging":
        from src.ensemble import alternative_ensemble
        from src.data import load_feature_label_pairs, ensure_data_dirs
        from src.data.preprocessing import infer_categorical_from_dtype
        from src.ensemble import find_best_models
        from src.utils.metrics import evaluate_model
        
        base_dir = Path(__file__).resolve().parent.parent
        raw_dir, _ = ensure_data_dirs(base_dir)
        results_dir = base_dir / "results"
        results_path = results_dir / "overfitting_experiments.json"
        
        print("=" * 60)
        print(f"앙상블 기법 실험: {args.method}")
        if args.method == "bagging":
            print(f"  n_bags: {args.n_bags}, sample_ratio: {args.bag_sample_ratio}")
        elif args.method == "boosting":
            print(f"  learning_rate: {args.learning_rate}")
        print("=" * 60)
        
        best_models = find_best_models(
            results_path,
            exclude_models=["NeuralNetwork", "DecisionTree", "RandomForest", "LightGBM", "Stacking", "StackingRF"],
            force_catboost_level1=True,
        )
        
        X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
        categorical_cols = infer_categorical_from_dtype(X_train, use_metadata=True)
        
        print(f"\n3. {args.method} 앙상블 예측 중...")
        if args.method == "bagging":
            ensemble_prob, weights = alternative_ensemble(
                best_models, X_train, y_train, X_test, 
                method=args.method, n_bags=args.n_bags, bag_sample_ratio=args.bag_sample_ratio
            )
        elif args.method == "boosting":
            ensemble_prob, weights = alternative_ensemble(
                best_models, X_train, y_train, X_test, 
                method=args.method, learning_rate=args.learning_rate
            )
        else:
            ensemble_prob, weights = alternative_ensemble(
                best_models, X_train, y_train, X_test, 
                method=args.method
            )
        
        print("\n  가중치:")
        for model_type, weight in weights.items():
            base_score = best_models[model_type]["test_score"]
            print(f"    {model_type}: {weight:.4f} (원래 Score: {base_score:.4f})")
        
        ensemble_metrics = evaluate_model(y_test, ensemble_prob)
        ensemble_score = (ensemble_metrics["f1"] + ensemble_metrics["auroc"]) / 2
        ensemble_metrics["score"] = ensemble_score
        
        print("\n" + "=" * 60)
        print("앙상블 결과")
        print("=" * 60)
        print(f"앙상블 Test Score: {ensemble_score:.4f}")
        print(f"  F1-score: {ensemble_metrics['f1']:.4f}")
        print(f"  AUROC: {ensemble_metrics['auroc']:.4f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "method": args.method,
            "best_models": best_models,
            "ensemble_result": ensemble_metrics,
            "weights": weights,
        }
        if args.method == "bagging":
            result_data["n_bags"] = args.n_bags
            result_data["bag_sample_ratio"] = args.bag_sample_ratio
        elif args.method == "boosting":
            result_data["learning_rate"] = args.learning_rate
        
        output_path = results_dir / f"alternative_ensemble_{args.method}_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {output_path}")
    else:
        main(method=args.method)

