# 최고 성능 분류 모델 파이프라인

EDA 결과를 기반으로 트리 기반 모델과 딥러닝 모델을 모두 구현하고 Optuna로 최적화하여 최고 성능의 분류 모델을 개발합니다.

## 프로젝트 구조

```
teamProject/
├── best_model_pipeline.py          # 전체 파이프라인 (메인 실행 파일)
├── models/
│   ├── __init__.py
│   ├── tree_models.py              # LightGBM, XGBoost, CatBoost
│   ├── neural_networks.py          # PyTorch 딥러닝 모델
│   └── ensemble.py                 # 앙상블 전략
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # 데이터 전처리
│   └── evaluation.py               # 평가 메트릭
├── optuna_studies/                 # Optuna 결과 저장
├── submissions/                    # 제출 파일 저장
├── requirements.txt                # 필요한 패키지
└── README_BEST_MODEL.md            # 이 파일
```

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. CUDA 환경 확인 (GPU 사용 시):
```bash
nvidia-smi
```

## 사용 방법

### 기본 실행

```bash
python best_model_pipeline.py
```

이 명령은 다음 작업을 수행합니다:

1. **데이터 전처리**
   - StandardScaler를 사용한 정규화
   - 상관관계가 높은 피처 제거 (상관계수 > 0.99)
   - Stratified K-Fold (5-fold) 교차 검증 설정

2. **트리 기반 모델 최적화**
   - LightGBM: 100 trials
   - XGBoost: 100 trials
   - CatBoost: 100 trials

3. **딥러닝 모델 최적화**
   - MLP (Multi-Layer Perceptron): 100 trials
   - Transformer: 50 trials

4. **모델 평가 및 선택**
   - 모든 모델의 5-fold CV 성능 비교
   - 최고 성능 모델 자동 선택

5. **최종 예측 및 제출**
   - 선택된 최고 모델로 test.csv 예측
   - submissions/ 디렉토리에 제출 파일 생성

### 결과 파일

- `submissions/submission_YYYYMMDD_HHMMSS.csv`: 제출 파일
- `models/preprocessor.pkl`: 저장된 전처리 파이프라인
- `models/best_model.*`: 최고 성능 모델 (형식은 모델 유형에 따라 다름)
- `models/results.pkl`: 최적화 결과 및 하이퍼파라미터

## 주요 기능

### 데이터 전처리

- **상관관계 기반 피처 제거**: 높은 상관관계(> 0.99)를 가진 중복 피처 자동 제거
- **정규화**: StandardScaler를 사용한 피처 스케일링
- **교차 검증**: Stratified K-Fold로 클래스 균형 유지

### 트리 기반 모델

- **LightGBM**: 학습률, 트리 깊이, 리프 수 등 최적화
- **XGBoost**: 감마, 정규화 계수 등 최적화
- **CatBoost**: 깊이, L2 정규화 등 최적화

### 딥러닝 모델

- **MLP**: 다층 퍼셉트론 기반 분류기
  - BatchNorm, Dropout 포함
  - AdamW 옵티마이저, CosineAnnealingLR 스케줄러
  - Mixed precision training (GPU 사용 시)
  
- **Transformer**: Tabular Transformer
  - Self-attention mechanism
  - Feature를 sequence로 변환하여 학습

### 하이퍼파라미터 최적화

- **Optuna**: 베이지안 최적화
- **Pruner**: MedianPruner로 성능 낮은 trial 조기 종료
- **Objective**: Macro-F1 Score 최대화

### 평가 지표

- **Primary**: Macro-F1 Score (대회 평가 지표)
- **Secondary**: Accuracy, Per-class F1-score
- **5-fold Cross-validation**: 평균 및 표준편차

## 모델 성능 비교

파이프라인 실행 후 자동으로 모든 모델의 성능을 비교하고, 최고 성능 모델을 선택합니다.

예시 출력:
```
모델 성능 비교
------------------------------------------------------------
1. lightgbm      : 0.950234 (+/- 0.001234)
2. xgboost       : 0.948567 (+/- 0.001567)
3. catboost      : 0.947123 (+/- 0.001890)
4. mlp            : 0.942345 (+/- 0.002345)
5. transformer    : 0.940123 (+/- 0.002678)

최고 성능 모델: lightgbm (Macro-F1: 0.950234)
```

## 커스터마이징

### 하이퍼파라미터 최적화 Trial 수 변경

`best_model_pipeline.py`의 `ModelPipeline` 초기화 시:

```python
pipeline = ModelPipeline(
    n_trials_tree=200,      # 트리 모델 trial 수 증가
    n_trials_nn=150,        # 딥러닝 모델 trial 수 증가
    n_trials_transformer=100
)
```

### 전처리 옵션 변경

`best_model_pipeline.py`의 `prepare_data` 메서드에서:

```python
self.preprocessor = DataPreprocessor(
    remove_high_corr=True,
    corr_threshold=0.95,     # 상관관계 임계값 조정
    use_feature_selection=True,
    top_n_features=40       # 선택할 피처 개수
)
```

## 주의사항

1. **실행 시간**: 전체 파이프라인 실행에 상당한 시간이 소요될 수 있습니다 (수 시간~수십 시간)
   - 트리 모델: 각 100 trials × 약 5-10분 = 약 25-50분
   - 딥러닝 모델: 각 100 trials × 약 10-20분 = 약 100-200분

2. **메모리**: 딥러닝 모델 학습 시 GPU 메모리가 필요합니다 (최소 8GB 권장)

3. **중간 결과 저장**: Optuna study는 자동으로 SQLite DB에 저장되어 재개 가능합니다.

## 문제 해결

### ImportError 발생 시

Python 경로 문제일 수 있습니다. 다음을 확인하세요:

```python
import sys
print(sys.path)
```

필요시 `best_model_pipeline.py`의 경로 설정 부분을 수정하세요.

### GPU 사용 불가 시

CPU로 자동 전환됩니다. 딥러닝 모델의 학습 시간이 길어질 수 있습니다.

### 메모리 부족 시

- Batch size를 줄이세요 (`neural_networks.py`에서)
- Trial 수를 줄이세요
- 한 번에 하나의 모델만 최적화하세요

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

