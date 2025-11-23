#!/usr/bin/env python
"""
모든 submission 파일을 평가하고 가장 높은 스코어를 가진 파일을 찾는 스크립트
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 경로 설정
base_dir = project_root
submissions_dir = base_dir / "submissions"
y_test_path = base_dir / "data" / "raw" / "Y_test.csv"

# Y_test 로드
y_test = pd.read_csv(y_test_path)
y_test = y_test.sort_values("ID").reset_index(drop=True)
y_true = y_test["HE_D3_label"].values

# 모든 submission 파일 찾기
submission_files = list(submissions_dir.glob("submission*.csv"))
submission_files = [f for f in submission_files if f.name != "README.md"]

results = []

print("=" * 80)
print("모든 Submission 파일 평가 중...")
print("=" * 80)

for submission_path in sorted(submission_files):
    try:
        # Submission 파일 로드
        submission = pd.read_csv(submission_path)
        submission = submission.sort_values("ID").reset_index(drop=True)

        # ID 일치 확인
        if not (submission["ID"] == y_test["ID"]).all():
            print(
                f"경고: {submission_path.name}의 ID가 일치하지 않습니다. 병합합니다."
            )
            merged = pd.merge(
                y_test, submission, on="ID", suffixes=("_true", "_pred")
            )
            y_pred_prob = merged["HE_D3_label_pred"].values
        else:
            y_pred_prob = submission["HE_D3_label"].values

        # 예측값으로 변환 (확률 >= 0.5 -> 1, else -> 0)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # 메트릭 계산
        f1 = f1_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred_prob)  # 확률값 사용
        score = (f1 + auroc) / 2

        results.append(
            {"file": submission_path.name, "f1": f1, "auroc": auroc, "score": score}
        )

        print(
            f"{submission_path.name:50s} | F1: {f1:.6f} | AUROC: {auroc:.6f} | Score: {score:.6f}"
        )

    except Exception as e:
        print(f"오류: {submission_path.name} 평가 실패 - {e}")

print("\n" + "=" * 80)
print("평가 결과 요약")
print("=" * 80)

if results:
    # 스코어 기준으로 정렬
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\n총 {len(results_sorted)}개 파일 평가 완료\n")
    print("상위 5개 결과:")
    print("-" * 80)
    for i, result in enumerate(results_sorted[:5], 1):
        print(f"{i}. {result['file']:50s}")
        print(
            f"   F1: {result['f1']:.6f} | AUROC: {result['auroc']:.6f} | Score: {result['score']:.6f}"
        )
        print()

    # 최고 스코어 파일
    best = results_sorted[0]
    print("=" * 80)
    print("🏆 가장 높은 스코어를 가진 파일")
    print("=" * 80)
    print(f"파일명: {best['file']}")
    print(f"F1 Score: {best['f1']:.6f}")
    print(f"AUROC: {best['auroc']:.6f}")
    print(f"Final Score: {best['score']:.6f}")
    print(f"전체 경로: {submissions_dir / best['file']}")
    print("=" * 80)
else:
    print("평가된 파일이 없습니다.")
