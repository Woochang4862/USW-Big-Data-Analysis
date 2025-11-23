#!/usr/bin/env python
"""
제출 파일 평가 스크립트
- submission.csv와 Y_test.csv를 비교하여 F1, AUROC, 최종 점수 계산
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def evaluate_submission(submission_path: Path, y_test_path: Path):
    """제출 파일 평가"""
    # CSV 파일 읽기
    sample_submission = pd.read_csv(submission_path)
    y_test = pd.read_csv(y_test_path)

    # ID를 기준으로 정렬하여 매칭
    sample_submission = sample_submission.sort_values("ID").reset_index(drop=True)
    y_test = y_test.sort_values("ID").reset_index(drop=True)

    # ID가 일치하는지 확인
    if not (sample_submission["ID"] == y_test["ID"]).all():
        print("경고: ID가 일치하지 않습니다. ID를 기준으로 병합합니다.")
        merged = pd.merge(
            y_test, sample_submission, on="ID", suffixes=("_true", "_pred")
        )
        y_true = merged["HE_D3_label_true"].values
        y_pred = merged["HE_D3_label_pred"].values
    else:
        y_true = y_test["HE_D3_label"].values
        y_pred = sample_submission["HE_D3_label"].values

    # F1 score 계산
    f1 = f1_score(y_true, y_pred)

    # AUROC 계산
    auroc = roc_auc_score(y_true, y_pred)

    # 최종 점수 계산
    score = (f1 + auroc) / 2

    # 결과 출력
    print("=" * 60)
    print("제출 파일 평가 결과")
    print("=" * 60)
    print(f"제출 파일: {submission_path}")
    print(f"F1 Score: {f1:.6f}")
    print(f"AUROC: {auroc:.6f}")
    print(f"Final Score (F1 + AUROC) / 2: {score:.6f}")
    print("=" * 60)

    return {"f1": f1, "auroc": auroc, "score": score}


def main():
    """메인 실행 함수"""
    base_dir = Path(__file__).resolve().parent.parent.parent

    # 기본 경로
    submission_path = base_dir / "submissions" / "submission.csv"
    y_test_path = base_dir / "data" / "raw" / "Y_test.csv"

    # 커맨드라인 인자 처리
    if len(sys.argv) > 1:
        submission_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        y_test_path = Path(sys.argv[2])

    if not submission_path.exists():
        print(f"오류: 제출 파일을 찾을 수 없습니다: {submission_path}")
        return

    if not y_test_path.exists():
        print(f"오류: 테스트 레이블 파일을 찾을 수 없습니다: {y_test_path}")
        return

    evaluate_submission(submission_path, y_test_path)


if __name__ == "__main__":
    main()
