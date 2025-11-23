# Final Assignment - ML 프로젝트

유지보수가 쉽고 확장성이 좋은 모듈화된 ML 프로젝트 구조입니다.

자세한 아키텍처 정보는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

## 사용 방법

### 1. 과적합 실험 실행

```bash
cd final-assignment
python experiments/overfitting_experiment.py
```

### 2. 앙상블 실험 실행

```bash
# 기본 가중합 앙상블
python experiments/ensemble_experiment.py

# 스태킹 앙상블
python experiments/ensemble_experiment.py --case stacking

# 특정 모델 제외
python experiments/ensemble_experiment.py --exclude DecisionTree NeuralNetwork
```

### 3. 결과 비교

```bash
# 모든 앙상블 방법 비교
python experiments/comparison/compare_all_ensemble_methods.py
```

### 4. 제출 파일 평가

```bash
# 기본 제출 파일 평가
python experiments/utils/test.py

# 특정 제출 파일 평가
python experiments/utils/test.py submissions/submission_20251118_151747.csv

# 모든 제출 파일 평가 및 최고 성능 파일 찾기
python experiments/utils/evaluate_all_submissions.py
```

### 5. 결과 시각화

```bash
python experiments/utils/visualize_results.py
```

### 6. 코드에서 모듈 사용

```python
from src.data import load_feature_label_pairs, impute_by_rules
from src.models.factory import create_model
from src.utils.metrics import evaluate_model

# 데이터 로드
X_train, y_train, X_test, y_test = load_feature_label_pairs(raw_dir)

# 모델 생성 및 학습
model = create_model("RandomForest", {"n_estimators": 100, "max_depth": 10})
model.fit(X_train, y_train)

# 예측 및 평가
y_prob = model.predict_proba(X_test)
metrics = evaluate_model(y_test, y_prob)
```

## 문서

- [ARCHITECTURE.md](ARCHITECTURE.md): 프로젝트 아키텍처 및 상세 구조 설명
- [submissions/README.md](submissions/README.md): 제출 파일 디렉토리 설명

## 라이선스

MIT
