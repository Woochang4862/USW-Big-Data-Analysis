#!/usr/bin/env python
"""
최고 성능 모델 앙상블 실험
- overfitting_experiments.json에서 각 모델 타입별 최고 test_score를 가진 하이퍼파라미터 선정
- 해당 하이퍼파라미터로 모델 재학습 및 앙상블 예측
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_feature_label_pairs, ensure_data_dirs
from src.data.preprocessing import infer_categorical_from_dtype, impute_by_rules
from src.ensemble import (
    stacking_ensemble,
    weighted_ensemble,
    find_best_models,
    get_model_by_rank,
)
from src.models.factory import create_model
from src.utils.metrics import evaluate_model


def main(
    case: Optional[str] = None,
    exclude_models: Optional[List[str]] = None,
    force_catboost_level1: bool = False,
    add_second_best: Optional[List[str]] = None,
):
    """메인 실행 함수

    Args:
        case: 케이스 이름 (결과 파일명에 사용)
        exclude_models: 제외할 모델 리스트
        force_catboost_level1: True이면 CatBoost는 Level1 강제 사용
        add_second_best: 추가로 포함할 모델 타입 리스트 (각 타입의 2nd best 추가)
    """
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir, _ = ensure_data_dirs(base_dir)
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
    if force_catboost_level1:
        print("   (CatBoost는 Level1 강제 사용)")
    best_models = find_best_models(
        results_path, exclude_models=exclude_models, force_catboost_level1=force_catboost_level1
    )

    if add_second_best:
        print("\n1-1. 추가 모델(2nd best) 선정 중...")
        for model_type in add_second_best:
            info = get_model_by_rank(results_path, model_type, rank=2)
            if info is None:
                print(f"  경고: {model_type}의 2번째 모델을 찾을 수 없습니다.")
                continue

            suffix = 2
            alias = f"{model_type}_rank{suffix}"
            while alias in best_models:
                suffix += 1
                alias = f"{model_type}_rank{suffix}"

            best_models[alias] = info
            print(
                f"  추가: {alias} (base: {model_type}) - "
                f"Config: {info['config_name']}, Test Score: {info['test_score']:.4f}"
            )

    for model_type, info in best_models.items():
        actual_model_type = info.get("model_type_actual", model_type)
        name_display = (
            model_type
            if model_type == actual_model_type
            else f"{model_type} (base: {actual_model_type})"
        )
        print(f"\n{name_display}:")
        print(f"  Config: {info['config_name']}")
        print(f"  Params: {info['params']}")
        print(
            f"  Test Score: {info['test_score']:.4f} "
            f"(F1: {info['test_f1']:.4f}, AUROC: {info['test_auroc']:.4f})"
        )

    # 데이터 로드 및 전처리
    print("\n2. 데이터 로드 및 전처리 중...")
    X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)
    categorical_cols = infer_categorical_from_dtype(X_train, use_metadata=True)
    X_train_imp, X_test_imp = impute_by_rules(X_train, X_test, categorical_cols)
    print(f"   Train: {X_train_imp.shape[0]} samples, {X_train_imp.shape[1]} features")
    print(f"   Test: {X_test_imp.shape[0]} samples")
    print(f"   감지된 범주형 변수: {len(categorical_cols)}개")

    # 개별 모델 평가
    print("\n3. 개별 모델 평가 중...")
    individual_results = {}

    for model_type, info in best_models.items():
        params = info["params"].copy()
        params["random_state"] = 42

        actual_model_type = info.get("model_type_actual", model_type)

        print(f"\n  {model_type} 학습 중...")
        if actual_model_type != model_type:
            print(f"    (모델 타입: {actual_model_type})")
        if actual_model_type == "CatBoost" and force_catboost_level1:
            print("    (CatBoost Level1 사용 - 범주형 변수 직접 활용)")
        model = create_model(actual_model_type, params)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)
        metrics = evaluate_model(y_test, y_prob)
        score = (metrics["f1"] + metrics["auroc"]) / 2
        metrics["score"] = score

        individual_results[model_type] = metrics
        print(
            f"  {model_type}: "
            f"F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}, Score={score:.4f}"
        )

    # 앙상블 방법 선택
    use_stacking = case and "stacking" in case.lower()

    if use_stacking and len(best_models) >= 2:
        # 스태킹 앙상블
        print("\n4. 스태킹 앙상블 예측 중...")
        ensemble_prob, meta_model, oof_predictions, test_predictions = stacking_ensemble(
            best_models, X_train, y_train, X_test, n_folds=5
        )

        # OOF 예측으로 메타 모델 평가
        meta_X_train = np.column_stack(
            [oof_predictions[mt] for mt in best_models.keys()]
        )
        oof_meta_pred = meta_model.predict_proba(meta_X_train)[:, 1]
        oof_metrics = evaluate_model(y_train, oof_meta_pred)
        oof_score = (oof_metrics["f1"] + oof_metrics["auroc"]) / 2

        print(f"\n  OOF 메타 모델 성능:")
        print(f"    Train Score: {oof_score:.4f}")
        print(f"    F1-score: {oof_metrics['f1']:.4f}")
        print(f"    AUROC: {oof_metrics['auroc']:.4f}")

        ensemble_metrics = evaluate_model(y_test, ensemble_prob)
        ensemble_score = (ensemble_metrics["f1"] + ensemble_metrics["auroc"]) / 2
        ensemble_metrics["score"] = ensemble_score
        normalized_weights = {}

    else:
        # 가중합 앙상블
        print("\n4. 가중합 앙상블 예측 중...")
        ensemble_prob, normalized_weights = weighted_ensemble(
            best_models, X_train, y_train, X_test
        )

        print("\n  가중치 (test_score^2 기반):")
        for model_type, weight in normalized_weights.items():
            base_score = best_models[model_type]["test_score"]
            print(
                f"    {model_type}: {weight:.4f} "
                f"(원래 Score: {base_score:.4f}, Score^2: {base_score**2:.4f})"
            )

        ensemble_metrics = evaluate_model(y_test, ensemble_prob)
        ensemble_score = (ensemble_metrics["f1"] + ensemble_metrics["auroc"]) / 2
        ensemble_metrics["score"] = ensemble_score
        meta_model = None
        oof_predictions = {}
        test_predictions = {}

    print("\n" + "=" * 60)
    print("앙상블 결과")
    print("=" * 60)
    print(f"앙상블 Test Score: {ensemble_score:.4f}")
    print(f"  F1-score: {ensemble_metrics['f1']:.4f}")
    print(f"  AUROC: {ensemble_metrics['auroc']:.4f}")

    # Submission 파일 생성
    print("\n5. Submission 파일 생성 중...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame(
        {"ID": X_test.index, "HE_D3_label": ensemble_prob}
    )
    submission_df = submission_df.set_index("ID")

    # submissions 디렉토리 생성
    submissions_dir = base_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    submission_path = submissions_dir / f"submission_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=True)
    print(f"Submission 파일 저장: {submission_path}")

    default_path = submissions_dir / "submission.csv"
    submission_df.to_csv(default_path, index=True)
    print(f"기본 Submission 파일 저장: {default_path}")

    # 결과 저장
    ensemble_result = {
        "best_models": best_models,
        "individual_results": individual_results,
        "ensemble_result": ensemble_metrics,
        "ensemble_method": "stacking" if use_stacking else "weighted_average",
        "weights": normalized_weights if not use_stacking else {},
    }

    if force_catboost_level1:
        ensemble_result["note"] = "CatBoost Level1 강제 사용 (범주형 변수 직접 활용)"

    if use_stacking:
        ensemble_result["meta_model"] = {"type": "LogisticRegression", "C": 1.0}

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
        print(
            f"{model_type:<20} {metrics['f1']:<8.4f} "
            f"{metrics['auroc']:<8.4f} {metrics['score']:<8.4f}"
        )
    print("-" * 60)
    print(
        f"{'Ensemble':<20} {ensemble_metrics['f1']:<8.4f} "
        f"{ensemble_metrics['auroc']:<8.4f} {ensemble_score:<8.4f}"
    )

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, help="케이스 이름")
    parser.add_argument("--exclude", nargs="+", help="제외할 모델 리스트")
    parser.add_argument(
        "--force-catboost-level1",
        action="store_true",
        help="CatBoost는 Level1 강제 사용 (범주형 변수 직접 활용)",
    )
    parser.add_argument(
        "--add-second-best",
        nargs="+",
        help="지정한 모델 타입의 2번째 최고 성능 모델을 추가로 포함",
    )
    args = parser.parse_args()

    exclude_models = None
    if args.exclude:
        exclude_models = args.exclude

    main(
        case=args.case,
        exclude_models=exclude_models,
        force_catboost_level1=args.force_catboost_level1,
        add_second_best=args.add_second_best,
    )

