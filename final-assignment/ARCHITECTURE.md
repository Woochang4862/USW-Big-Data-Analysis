# 프로젝트 아키텍처

유지보수가 쉽고 확장성이 좋은 모듈화된 ML 프로젝트 구조입니다.

## 프로젝트 구조

```
final-assignment/
├── src/                          # 핵심 소스 코드 모듈
│   ├── data/                    # 데이터 로드 및 전처리
│   ├── models/                  # 모델 정의
│   ├── ensemble/                # 앙상블 모듈
│   └── utils/                   # 유틸리티 함수
│
├── experiments/                  # 실험 스크립트
│   ├── overfitting_experiment.py      # 과적합 실험
│   ├── ensemble_experiment.py         # 앙상블 실험
│   ├── comparison/                    # 비교 스크립트
│   │   ├── compare_all_ensemble_methods.py
│   │   ├── compare_ensemble_cases.py
│   │   ├── compare_ensemble_with_without_dt.py
│   │   └── compare_stacking_vs_weighted.py
│   └── utils/                         # 유틸리티 스크립트
│       ├── test.py                    # 제출 파일 평가
│       ├── evaluate_all_submissions.py # 모든 제출 파일 평가 및 최고 성능 파일 찾기
│       └── visualize_results.py       # 결과 시각화
│
├── submissions/                 # 제출 파일들
│   ├── submission.csv          # 최신 제출 파일
│   └── submission_*.csv        # 이전 제출 파일들
│
├── legacy/                      # 기존 파일들 (참고용)
│   ├── modeling.py
│   ├── overfitting_experiment.py
│   └── ensemble_best_models.py
│
├── config/                      # 설정 파일
├── data/                       # 데이터 디렉토리
├── results/                    # 실험 결과
└── docs/                       # 문서
```

## 주요 특징

### 1. 모듈화된 구조
- **데이터 모듈**: 데이터 로드 및 전처리 로직을 재사용 가능한 모듈로 분리
- **모델 모듈**: 각 모델을 독립적인 클래스로 구현, 팩토리 패턴으로 통일된 인터페이스 제공
- **앙상블 모듈**: 스태킹과 가중합 앙상블을 별도 모듈로 분리
- **유틸리티 모듈**: 공통 함수들을 중앙화

### 2. 확장성
- 새로운 모델 추가: `src/models/`에 새 모델 클래스 추가 후 `factory.py`에 등록
- 새로운 전처리 방법: `src/data/preprocessing.py`에 함수 추가
- 새로운 앙상블 방법: `src/ensemble/`에 새 모듈 추가

### 3. 유지보수성
- 코드 중복 제거: 공통 로직을 모듈로 분리
- 명확한 책임 분리: 각 모듈이 단일 책임을 가짐
- 타입 힌팅: 모든 함수에 타입 힌팅 적용

## 파일 분류 기준

### src/에 포함되는 파일
- 재사용 가능한 모듈
- 다른 스크립트에서 import하여 사용하는 코드
- 공통 기능 (데이터 로드, 전처리, 모델 정의, 평가 메트릭 등)

### experiments/에 포함되는 파일
- 실행 가능한 스크립트
- 실험 실행 및 결과 분석
- 비교 및 시각화 스크립트

**주요 스크립트:**
- `overfitting_experiment.py`: 과적합 실험 실행
- `ensemble_experiment.py`: 앙상블 실험 실행
- `comparison/compare_all_ensemble_methods.py`: 모든 앙상블 방법 비교
- `comparison/compare_ensemble_cases.py`: 앙상블 케이스 비교
- `comparison/compare_ensemble_with_without_dt.py`: Decision Tree 포함/제외 비교
- `comparison/compare_stacking_vs_weighted.py`: 스태킹 vs 가중합 비교
- `utils/test.py`: 제출 파일 평가 (F1, AUROC 계산)
- `utils/evaluate_all_submissions.py`: 모든 제출 파일 평가 및 최고 성능 파일 찾기
- `utils/visualize_results.py`: 실험 결과 시각화

### submissions/에 포함되는 파일
- 제출용 CSV 파일
- 모델 예측 결과

### legacy/에 포함되는 파일
- 기존 버전의 스크립트
- 참고용으로 보관
- 새로운 구조로 대체됨
- **주의**: 이 디렉토리의 파일은 수정하지 마세요

## 모듈 사용 방법

### 데이터 모듈
```python
from src.data import load_feature_label_pairs, impute_by_rules

# 데이터 로드
X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)

# 전처리
X_train_processed = impute_by_rules(X_train)
```

### 모델 모듈
```python
from src.models.factory import create_model

# 모델 생성 및 학습
model = create_model("RandomForest", {"n_estimators": 100, "max_depth": 10})
model.fit(X_train, y_train)

# 예측
y_prob = model.predict_proba(X_test)
```

### 평가 모듈
```python
from src.utils.metrics import evaluate_model

# 평가
metrics = evaluate_model(y_test, y_prob)
```

### 앙상블 모듈
```python
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.weighted import WeightedEnsemble

# 스태킹 앙상블
stacking = StackingEnsemble(base_models, meta_model)
stacking.fit(X_train, y_train)
y_pred = stacking.predict_proba(X_test)
```

## 모델 추가 방법

1. `src/models/`에 새 모델 클래스 생성 (예: `my_model.py`)
2. `BaseModel`을 상속받아 구현
3. `src/models/factory.py`의 `ModelFactory._model_classes`에 등록

예시:
```python
# src/models/my_model.py
from src.models.base import BaseModel

class MyModel(BaseModel):
    def fit(self, X_train, y_train):
        # 학습 로직
        return self
    
    def predict_proba(self, X):
        # 예측 로직
        return probabilities
```

## 리팩토링 마이그레이션 가이드

프로젝트가 모듈화된 구조로 리팩토링되었습니다. 기존 파일들은 그대로 유지되며, 새로운 구조를 사용하는 것을 권장합니다.

### 주요 변경사항

#### 1. 디렉토리 구조
- **새로운 구조**: `src/`, `experiments/`, `config/` 디렉토리 추가
- **기존 파일**: 루트 디렉토리에 유지 (하위 호환성)

#### 2. 모듈화
- 데이터 로드/전처리: `src/data/`
- 모델 정의: `src/models/`
- 앙상블: `src/ensemble/`
- 유틸리티: `src/utils/`

#### 3. 실험 스크립트
- `overfitting_experiment.py` → `experiments/overfitting_experiment.py`
- `ensemble_best_models.py` → `experiments/ensemble_experiment.py`

### 마이그레이션 방법

#### 기존 코드 사용 시
기존 스크립트들은 그대로 작동합니다. 다만 새로운 모듈을 사용하도록 업데이트하는 것을 권장합니다.

#### 새로운 구조 사용 시

```python
# 기존 방식
from overfitting_experiment import load_feature_label_pairs

# 새로운 방식
from src.data import load_feature_label_pairs
```

### 파일 매핑

| 기존 파일 | 새로운 위치 | 비고 |
|---------|-----------|------|
| `modeling.py` | `experiments/` (새로 작성 필요) | 기본 모델링 스크립트 |
| `overfitting_experiment.py` | `experiments/overfitting_experiment.py` | 리팩토링됨 |
| `ensemble_best_models.py` | `experiments/ensemble_experiment.py` | 리팩토링됨 |
| `compare_*.py` | `experiments/comparison/` (유지) | 비교 스크립트 |
| `visualize_results.py` | `experiments/utils/` (유지) | 시각화 스크립트 |

## 의존성

주요 패키지:
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow (선택사항, Neural Network 사용 시)

