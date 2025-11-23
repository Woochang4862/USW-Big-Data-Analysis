# 제출 파일 디렉토리

이 디렉토리에는 모델 예측 결과 제출 파일들이 저장됩니다.

## 파일 설명

- **submission.csv**: 최신 제출 파일 (기본 제출 파일)
- **submission_YYYYMMDD_HHMMSS.csv**: 타임스탬프가 포함된 제출 파일들

## 제출 파일 평가

### 단일 파일 평가

제출 파일의 성능을 평가하려면:

```bash
# 기본 제출 파일 평가
python experiments/utils/test.py

# 특정 제출 파일 평가
python experiments/utils/test.py submissions/submission_20251118_151747.csv
```

### 모든 파일 평가 및 최고 성능 파일 찾기

모든 제출 파일을 한 번에 평가하고 가장 높은 스코어를 가진 파일을 찾으려면:

```bash
python experiments/utils/evaluate_all_submissions.py
```

이 스크립트는:
- `submissions/` 디렉토리의 모든 `submission*.csv` 파일을 평가
- 각 파일의 F1 Score, AUROC, Final Score를 계산
- 상위 5개 결과를 표시
- 가장 높은 스코어를 가진 파일을 강조 표시

### 직접 평가

```python
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# 제출 파일과 실제 레이블 로드
submission = pd.read_csv('submissions/submission.csv')
y_test = pd.read_csv('data/raw/Y_test.csv')

# 평가
f1 = f1_score(y_test['HE_D3_label'], submission['HE_D3_label'])
auroc = roc_auc_score(y_test['HE_D3_label'], submission['HE_D3_label'])
score = (f1 + auroc) / 2

print(f"F1 Score: {f1:.6f}")
print(f"AUROC: {auroc:.6f}")
print(f"Final Score: {score:.6f}")
```

