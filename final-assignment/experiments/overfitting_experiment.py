#!/usr/bin/env python
"""
모델 과적합 실험 파이프라인
- raw 데이터를 사용하여 간단한 모델부터 복잡한 모델까지 점진적으로 과적합 실험
- Train/Test 성능 차이를 통해 과적합 정도 측정
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_feature_label_pairs, ensure_data_dirs
from src.data.preprocessing import infer_categorical_from_dtype, impute_by_rules
from src.models.factory import create_model
from src.utils.metrics import evaluate_model, build_result
from src.ensemble import stacking_ensemble, find_best_models


def get_model_configs(model_type: str) -> List[Dict]:
    """모델 타입별 실험 설정 반환"""
    configs = {
        "LogisticRegression": [
            {"name": "Level-10", "C": 100000.0},
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
            {"name": "Level14", "C": 0.00001},
        ],
        "DecisionTree": [
            {"name": "Level1", "max_depth": 1},
            {"name": "Level2", "max_depth": 5},
            {"name": "Level3", "max_depth": 15},
            {"name": "Level4", "max_depth": 50},
            {"name": "Level5", "max_depth": None},
        ],
        "RandomForest": [
            {"name": "Level1", "n_estimators": 10, "max_depth": 5},
            {"name": "Level2", "n_estimators": 50, "max_depth": 10},
            {"name": "Level3", "n_estimators": 100, "max_depth": 15},
            {"name": "Level4", "n_estimators": 500, "max_depth": 30},
            {"name": "Level5", "n_estimators": 1000, "max_depth": None},
        ],
        "XGBoost": [
            {"name": "Level1", "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
            {"name": "Level2", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
            {"name": "Level3", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            {"name": "Level4", "n_estimators": 500, "max_depth": 10, "learning_rate": 0.1},
            {"name": "Level5", "n_estimators": 1000, "max_depth": 15, "learning_rate": 0.3},
        ],
        "LightGBM": [
            {"name": "Level-10", "n_estimators": 1, "max_depth": 1, "learning_rate": 0.1, "num_leaves": 2},
            {"name": "Level-9", "n_estimators": 2, "max_depth": 1, "learning_rate": 0.1, "num_leaves": 2},
            {"name": "Level-8", "n_estimators": 3, "max_depth": 1, "learning_rate": 0.1, "num_leaves": 3},
            {"name": "Level-7", "n_estimators": 5, "max_depth": 1, "learning_rate": 0.1, "num_leaves": 3},
            {"name": "Level-6", "n_estimators": 7, "max_depth": 2, "learning_rate": 0.1, "num_leaves": 4},
            {"name": "Level-5", "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "num_leaves": 7},
            {"name": "Level-4", "n_estimators": 15, "max_depth": 2, "learning_rate": 0.1, "num_leaves": 7},
            {"name": "Level-3", "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "num_leaves": 8},
            {"name": "Level-2", "n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "num_leaves": 15},
            {"name": "Level-1", "n_estimators": 40, "max_depth": 4, "learning_rate": 0.1, "num_leaves": 16},
            {"name": "Level0", "n_estimators": 50, "max_depth": 5, "learning_rate": 0.1, "num_leaves": 31},
            {"name": "Level1", "n_estimators": 80, "max_depth": 6, "learning_rate": 0.1, "num_leaves": 63},
            {"name": "Level2", "n_estimators": 100, "max_depth": 7, "learning_rate": 0.1, "num_leaves": 127},
            {"name": "Level3", "n_estimators": 150, "max_depth": 9, "learning_rate": 0.1, "num_leaves": 511},
            {"name": "Level4", "n_estimators": 200, "max_depth": 11, "learning_rate": 0.1, "num_leaves": 2047},
            {"name": "Level5", "n_estimators": 300, "max_depth": 14, "learning_rate": 0.1, "num_leaves": 16383},
            {"name": "Level6", "n_estimators": 500, "max_depth": 18, "learning_rate": 0.1, "num_leaves": 65535},
            {"name": "Level7", "n_estimators": 700, "max_depth": 22, "learning_rate": 0.1, "num_leaves": 131072},
            {"name": "Level8", "n_estimators": 1000, "max_depth": 25, "learning_rate": 0.1, "num_leaves": 131072},
            {"name": "Level9", "n_estimators": 1500, "max_depth": 28, "learning_rate": 0.2, "num_leaves": 131072},
            {"name": "Level10", "n_estimators": 2000, "max_depth": 30, "learning_rate": 0.3, "num_leaves": 131072},
            {"name": "Level11", "n_estimators": 2500, "max_depth": 30, "learning_rate": 0.4, "num_leaves": 131072},
            {"name": "Level12", "n_estimators": 3000, "max_depth": 30, "learning_rate": 0.5, "num_leaves": 131072},
            {"name": "Level13", "n_estimators": 4000, "max_depth": 30, "learning_rate": 0.5, "num_leaves": 131072},
            {"name": "Level14", "n_estimators": 5000, "max_depth": 30, "learning_rate": 0.5, "num_leaves": 131072},
        ],
        "NeuralNetwork": [
            {"name": "Level1", "layers": [64], "epochs": 50},
            {"name": "Level2", "layers": [128, 64], "epochs": 50},
            {"name": "Level3", "layers": [256, 128, 64], "epochs": 50},
            {"name": "Level4", "layers": [512, 256, 128, 64], "epochs": 100},
            {"name": "Level5", "layers": [512, 256, 128, 64, 32], "epochs": 200},
        ],
        "CatBoost": [
            {"name": "Level-10", "iterations": 1, "depth": 1, "learning_rate": 0.1, "l2_leaf_reg": 100.0},
            {"name": "Level-9", "iterations": 2, "depth": 1, "learning_rate": 0.1, "l2_leaf_reg": 50.0},
            {"name": "Level-8", "iterations": 3, "depth": 1, "learning_rate": 0.1, "l2_leaf_reg": 30.0},
            {"name": "Level-7", "iterations": 5, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 20.0},
            {"name": "Level-6", "iterations": 7, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 15.0},
            {"name": "Level-5", "iterations": 10, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 10.0},
            {"name": "Level-4", "iterations": 15, "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 8.0},
            {"name": "Level-3", "iterations": 20, "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 5.0},
            {"name": "Level-2", "iterations": 30, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
            {"name": "Level-1", "iterations": 40, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 2.0},
            {"name": "Level0", "iterations": 50, "depth": 5, "learning_rate": 0.1, "l2_leaf_reg": 1.0},
            {"name": "Level1", "iterations": 80, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 0.5},
            {"name": "Level2", "iterations": 100, "depth": 7, "learning_rate": 0.1, "l2_leaf_reg": 0.3},
            {"name": "Level3", "iterations": 150, "depth": 8, "learning_rate": 0.1, "l2_leaf_reg": 0.1},
            {"name": "Level4", "iterations": 200, "depth": 9, "learning_rate": 0.1, "l2_leaf_reg": 0.05},
            # {"name": "Level5", "iterations": 300, "depth": 10, "learning_rate": 0.1, "l2_leaf_reg": 0.01},
            # {"name": "Level6", "iterations": 500, "depth": 12, "learning_rate": 0.1, "l2_leaf_reg": 0.005},
            # {"name": "Level7", "iterations": 700, "depth": 14, "learning_rate": 0.1, "l2_leaf_reg": 0.001},
            # {"name": "Level8", "iterations": 1000, "depth": 16, "learning_rate": 0.1, "l2_leaf_reg": 0.0005},
            # {"name": "Level9", "iterations": 1500, "depth": 16, "learning_rate": 0.2, "l2_leaf_reg": 0.0001},
            # {"name": "Level10", "iterations": 2000, "depth": 16, "learning_rate": 0.3, "l2_leaf_reg": 0.00005},
            # {"name": "Level11", "iterations": 2500, "depth": 16, "learning_rate": 0.4, "l2_leaf_reg": 0.00001},
            # {"name": "Level12", "iterations": 3000, "depth": 16, "learning_rate": 0.5, "l2_leaf_reg": 0.000005},
            # {"name": "Level13", "iterations": 4000, "depth": 16, "learning_rate": 0.5, "l2_leaf_reg": 0.000001},
            # {"name": "Level14", "iterations": 5000, "depth": 16, "learning_rate": 0.5, "l2_leaf_reg": 0.0000001},
        ],
        "Stacking": [
            {"name": "Level-10", "meta_C": 0.00001},
            {"name": "Level-9", "meta_C": 0.00005},
            {"name": "Level-8", "meta_C": 0.0001},
            {"name": "Level-7", "meta_C": 0.0005},
            {"name": "Level-6", "meta_C": 0.001},
            {"name": "Level-5", "meta_C": 0.005},
            {"name": "Level-4", "meta_C": 0.01},
            {"name": "Level-3", "meta_C": 0.05},
            {"name": "Level-2", "meta_C": 0.1},
            {"name": "Level-1", "meta_C": 0.5},
            {"name": "Level0", "meta_C": 1.0},
            {"name": "Level1", "meta_C": 5.0},
            {"name": "Level2", "meta_C": 10.0},
            {"name": "Level3", "meta_C": 50.0},
            {"name": "Level4", "meta_C": 100.0},
            {"name": "Level5", "meta_C": 500.0},
            {"name": "Level6", "meta_C": 1000.0},
            {"name": "Level7", "meta_C": 5000.0},
            {"name": "Level8", "meta_C": 10000.0},
            {"name": "Level9", "meta_C": 50000.0},
            {"name": "Level10", "meta_C": 100000.0},
            {"name": "Level11", "meta_C": 500000.0},
            {"name": "Level12", "meta_C": 1000000.0},
            {"name": "Level13", "meta_C": 5000000.0},
            {"name": "Level14", "meta_C": 10000000.0},
        ],
        "StackingRF": [
            {"name": "Level-10", "n_estimators": 1, "max_depth": 1},
            {"name": "Level-9", "n_estimators": 2, "max_depth": 1},
            {"name": "Level-8", "n_estimators": 3, "max_depth": 2},
            {"name": "Level-7", "n_estimators": 5, "max_depth": 2},
            {"name": "Level-6", "n_estimators": 7, "max_depth": 3},
            {"name": "Level-5", "n_estimators": 10, "max_depth": 3},
            {"name": "Level-4", "n_estimators": 15, "max_depth": 4},
            {"name": "Level-3", "n_estimators": 20, "max_depth": 4},
            {"name": "Level-2", "n_estimators": 30, "max_depth": 5},
            {"name": "Level-1", "n_estimators": 40, "max_depth": 5},
            {"name": "Level0", "n_estimators": 50, "max_depth": 6},
            {"name": "Level1", "n_estimators": 70, "max_depth": 7},
            {"name": "Level2", "n_estimators": 100, "max_depth": 8},
            {"name": "Level3", "n_estimators": 150, "max_depth": 10},
            {"name": "Level4", "n_estimators": 200, "max_depth": 12},
            {"name": "Level5", "n_estimators": 300, "max_depth": 15},
            {"name": "Level6", "n_estimators": 400, "max_depth": 18},
            {"name": "Level7", "n_estimators": 500, "max_depth": 20},
            {"name": "Level8", "n_estimators": 600, "max_depth": 22},
            {"name": "Level9", "n_estimators": 700, "max_depth": 25},
            {"name": "Level10", "n_estimators": 800, "max_depth": 27},
            {"name": "Level11", "n_estimators": 900, "max_depth": 28},
            {"name": "Level12", "n_estimators": 1000, "max_depth": 30},
            {"name": "Level13", "n_estimators": 1200, "max_depth": 30},
            {"name": "Level14", "n_estimators": 1500, "max_depth": 30},
        ],
    }
    return configs.get(model_type, [])


def run_experiment(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_path: Optional[Path] = None,
) -> List[Dict]:
    """단일 모델 타입에 대한 실험 실행"""
    results = []
    configs = get_model_configs(model_type)

    if not configs:
        print(f"  경고: {model_type}에 대한 설정이 없습니다.")
        return results

    # Stacking 모델의 경우 base 모델들을 찾아야 함
    if model_type in ["Stacking", "StackingRF"]:
        if results_path is None:
            print("  경고: Stacking 실험을 위해서는 results_path가 필요합니다.")
            return results
        
        # Base 모델 찾기 (3개 모델: LogisticRegression, XGBoost, CatBoost)
        exclude_list = ["NeuralNetwork", "DecisionTree", "RandomForest", "LightGBM", "Stacking", "StackingRF"]
        best_models = find_best_models(
            results_path,
            exclude_models=exclude_list,
            force_catboost_level1=True,
        )
        
        if len(best_models) < 2:
            print("  경고: Stacking을 위한 base 모델이 부족합니다.")
            return results
        
        print(f"  Base 모델: {', '.join(best_models.keys())}")

    for config in configs:
        if model_type == "Stacking":
            # Stacking 실험 (LogisticRegression 메타 모델)
            meta_C = config["meta_C"]
            try:
                # 스태킹 앙상블 실행
                ensemble_prob, meta_model, oof_predictions, test_predictions = stacking_ensemble(
                    best_models, X_train, y_train, X_test, n_folds=5, meta_C=meta_C
                )
                
                # Train 예측 (OOF 예측을 메타 피처로 사용)
                import numpy as np
                meta_X_train = np.column_stack([oof_predictions[mt] for mt in best_models.keys()])
                train_prob = meta_model.predict_proba(meta_X_train)[:, 1]
                test_prob = ensemble_prob

                train_metrics = evaluate_model(y_train, train_prob)
                test_metrics = evaluate_model(y_test, test_prob)

                params = {"meta_C": meta_C, "base_models": list(best_models.keys())}
                result = build_result(
                    model_type=model_type,
                    config_name=config["name"],
                    params=params,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                )
                results.append(result)

                print(
                    f"  {config['name']}: "
                    f"Train Score={result['train_score']:.4f} | "
                    f"Test Score={result['test_score']:.4f} | "
                    f"Gap={result['overfitting_gap_score']:.4f}"
                )
            except Exception as e:
                print(f"  {config['name']}: 실패 - {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        elif model_type == "StackingRF":
            # StackingRF 실험 (RandomForest 메타 모델)
            meta_params = {k: v for k, v in config.items() if k != "name"}
            try:
                # 스태킹 앙상블 실행
                ensemble_prob, meta_model, oof_predictions, test_predictions = stacking_ensemble(
                    best_models, X_train, y_train, X_test, n_folds=5,
                    meta_model_type="RandomForest",
                    meta_params=meta_params
                )
                
                # Train 예측 (OOF 예측을 메타 피처로 사용)
                import numpy as np
                meta_X_train = np.column_stack([oof_predictions[mt] for mt in best_models.keys()])
                train_prob = meta_model.predict_proba(meta_X_train)[:, 1]
                test_prob = ensemble_prob

                train_metrics = evaluate_model(y_train, train_prob)
                test_metrics = evaluate_model(y_test, test_prob)

                params = {"meta_model_type": "RandomForest", "meta_params": meta_params, "base_models": list(best_models.keys())}
                result = build_result(
                    model_type=model_type,
                    config_name=config["name"],
                    params=params,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                )
                results.append(result)

                print(
                    f"  {config['name']}: "
                    f"Train Score={result['train_score']:.4f} | "
                    f"Test Score={result['test_score']:.4f} | "
                    f"Gap={result['overfitting_gap_score']:.4f}"
                )
            except Exception as e:
                print(f"  {config['name']}: 실패 - {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        else:
            # 일반 모델 실험
            params = {k: v for k, v in config.items() if k != "name"}
            params["random_state"] = 42

            try:
                model = create_model(model_type, params)
                model.fit(X_train, y_train)

                train_prob = model.predict_proba(X_train)
                test_prob = model.predict_proba(X_test)

                train_metrics = evaluate_model(y_train, train_prob)
                test_metrics = evaluate_model(y_test, test_prob)

                result = build_result(
                    model_type=model_type,
                    config_name=config["name"],
                    params=params,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                )
                results.append(result)

                print(
                    f"  {config['name']}: "
                    f"Train Score={result['train_score']:.4f} | "
                    f"Test Score={result['test_score']:.4f} | "
                    f"Gap={result['overfitting_gap_score']:.4f}"
                )
            except Exception as e:
                print(f"  {config['name']}: 실패 - {str(e)}")
                continue

    return results


def load_existing_results(results_dir: Path) -> List[Dict]:
    """기존 결과 로드"""
    json_path = results_dir / "overfitting_experiments.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(
    all_results: List[Dict],
    results_dir: Path,
    update_models: Optional[List[str]] = None,
) -> None:
    """결과를 JSON과 CSV로 저장"""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 기존 결과 로드 (선택적 업데이트인 경우)
    if update_models is not None:
        existing_results = load_existing_results(results_dir)
        existing_results = [
            r for r in existing_results if r.get("model_type") not in update_models
        ]
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
        summary_data.append(
            {
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
            }
        )

    summary_df = pd.DataFrame(summary_data)
    csv_path = results_dir / "overfitting_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"요약 저장: {csv_path}")

    # 요약 출력
    print("\n=== 실험 요약 ===")
    print(summary_df.to_string(index=False))


def main():
    """메인 실행 함수"""
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir, _ = ensure_data_dirs(base_dir)
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
    categorical_cols = infer_categorical_from_dtype(X_train, use_metadata=True)
    print(f"   범주형 변수: {len(categorical_cols)}개 (메타데이터 포함)")
    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)
    print("   결측치 처리 완료")

    # 실험 실행
    all_results = []
    results_path = results_dir / "overfitting_experiments.json"
    model_types = ["StackingRF"]

    for i, model_type in enumerate(model_types, 3):
        print(f"\n{i}. {model_type} 실험 중...")
        if model_type == "Stacking":
            print(f"   (메타 모델: LogisticRegression, 복잡도 조절: Level-10~14)")
        elif model_type == "StackingRF":
            print(f"   (메타 모델: RandomForest, 복잡도 조절: Level-10~14)")
        try:
            results = run_experiment(
                model_type, X_train, y_train, X_test, y_test, results_path=results_path
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  경고: {model_type} 실험 실패 - {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 결과 저장 (기존 결과에 StackingRF 추가)
    print("\n4. 결과 저장 중...")
    save_results(all_results, results_dir, update_models=["StackingRF"])

    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

