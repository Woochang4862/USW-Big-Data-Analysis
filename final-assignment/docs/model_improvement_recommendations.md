# 모델 개선 방안

## 현재 모델 성능 분석 결과

### 주요 통계
- **전체 정확도**: 40.17% (매우 낮음)
- **정확한 예측**: 1,493개 / 2,173개
- **오류 유형**:
  - **False Positive (실제=0, 예측=1)**: 488개 (오류의 71.5%)
  - **False Negative (실제=1, 예측=0)**: 192개 (오류의 28.5%)
- **최적 임계값**: 0.35 (현재 0.5 사용 중)

### 오류 패턴 분석
1. **False Positive (FP) 문제가 심각**
   - 평균 예측 확률: 0.6853
   - 중앙값: 0.6907
   - 모델이 실제로는 0인데 1로 예측하는 경우가 많음
   - 높은 확률(0.6~0.8)로 잘못 예측하는 경우가 많음

2. **False Negative (FN) 문제**
   - 평균 예측 확률: 0.3586
   - 중앙값: 0.3777
   - 모델이 실제로는 1인데 0으로 예측
   - 낮은 확률(0.2~0.5)로 예측하는 경우가 많음

## 개선 방안

### 1. 임계값 최적화 (즉시 적용 가능) ⭐⭐⭐

**문제**: 현재 임계값 0.5를 사용하지만, 최적 임계값은 0.35입니다.

**해결책**:
```python
# submission 생성 시 임계값 조정
threshold = 0.35  # 또는 0.3~0.4 범위에서 최적값 탐색
pred_binary = (pred_proba >= threshold).astype(int)
```

**예상 효과**: F1 점수 0.774 (현재보다 약 0.6% 향상)

**구현 위치**: `final-assignment/experiments/ensemble_experiment.py` 또는 submission 생성 코드

---

### 2. 클래스 불균형 처리 ⭐⭐⭐

**문제**: False Positive가 False Negative보다 2.5배 많음. 클래스 불균형 가능성.

**해결책 A: 클래스 가중치 조정**
```python
# LogisticRegression
model = LogisticRegression(
    class_weight='balanced',  # 또는 {0: 0.6, 1: 0.4} 같은 커스텀 가중치
    ...
)

# XGBoost
model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    ...
)

# RandomForest
model = RandomForestClassifier(
    class_weight='balanced',
    ...
)
```

**해결책 B: SMOTE 오버샘플링**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**예상 효과**: False Positive 감소, 전체 정확도 향상

---

### 3. 앙상블 가중치 재조정 ⭐⭐

**문제**: 현재 가중합 앙상블이 False Positive를 과도하게 생성

**해결책 A: 오류 유형별 성능을 고려한 가중치**
```python
# False Positive를 줄이는 모델에 더 높은 가중치 부여
# 각 모델의 Precision을 고려한 가중치 계산

def calculate_precision_weighted_weights(models_performance):
    """Precision을 고려한 가중치 계산"""
    weights = {}
    total_precision = sum(perf['precision'] for perf in models_performance.values())
    
    for model_name, perf in models_performance.items():
        # Precision이 높을수록 False Positive가 적음
        weights[model_name] = perf['precision'] / total_precision
    
    return weights
```

**해결책 B: 오류 유형별 가중치**
```python
# False Positive를 줄이는 것이 목표이므로
# Precision에 더 높은 가중치를 부여
weights = {
    'LogisticRegression': 0.3,  # Precision이 높은 모델
    'XGBoost': 0.4,
    'RandomForest': 0.3
}
```

---

### 4. 모델 하이퍼파라미터 튜닝 ⭐⭐

**문제**: 현재 모델들이 False Positive를 과도하게 생성

**해결책 A: LogisticRegression - C 값 조정**
```python
# 더 보수적인 예측을 위해 C 값을 낮춤
params = {
    'C': 0.001,  # 현재 0.01에서 더 낮춤
    'class_weight': 'balanced'
}
```

**해결책 B: XGBoost - 정규화 강화**
```python
params = {
    'n_estimators': 50,  # 현재 20에서 증가
    'max_depth': 2,  # 현재 3에서 감소 (과적합 방지)
    'learning_rate': 0.05,  # 현재 0.1에서 감소
    'min_child_weight': 5,  # 추가: 더 보수적인 분할
    'reg_alpha': 0.1,  # L1 정규화
    'reg_lambda': 0.1,  # L2 정규화
    'scale_pos_weight': 1.5  # 클래스 불균형 처리
}
```

**해결책 C: RandomForest - 더 보수적인 설정**
```python
params = {
    'n_estimators': 300,  # 현재 200에서 증가
    'max_depth': 10,  # 현재 15에서 감소
    'min_samples_split': 10,  # 추가: 더 보수적인 분할
    'min_samples_leaf': 5,  # 추가
    'class_weight': 'balanced'
}
```

---

### 5. 새로운 모델 추가 ⭐

**추천 모델**:
1. **CatBoost**: 범주형 변수 처리에 강함
2. **LightGBM**: 이미 사용 중이지만 파라미터 튜닝 필요
3. **Neural Network**: 비선형 패턴 학습

**구현 예시**:
```python
from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='F1',
    class_weights=[0.6, 0.4],  # 클래스 불균형 처리
    random_seed=42,
    verbose=False
)
```

---

### 6. 피처 엔지니어링 ⭐⭐

**문제**: 현재 피처가 모델의 판단에 충분하지 않을 수 있음

**해결책 A: 상호작용 피처 추가**
```python
# 중요한 피처 간 상호작용 생성
X_train['feature1_x_feature2'] = X_train['feature1'] * X_train['feature2']
X_train['feature1_div_feature2'] = X_train['feature1'] / (X_train['feature2'] + 1e-8)
```

**해결책 B: 통계적 피처**
```python
# 그룹별 통계 피처
X_train['mean_by_group'] = X_train.groupby('group_col')['value_col'].transform('mean')
X_train['std_by_group'] = X_train.groupby('group_col')['value_col'].transform('std')
```

**해결책 C: 피처 선택**
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=50)  # 상위 50개 피처만 선택
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

---

### 7. 교차 검증 기반 앙상블 개선 ⭐⭐

**문제**: 현재 앙상블이 단일 예측에 의존

**해결책: 다중 교차 검증 앙상블**
```python
# 여러 교차 검증 폴드에서 예측을 평균
n_folds = 5
predictions = []

for fold in range(n_folds):
    # 각 폴드에서 모델 학습 및 예측
    model.fit(X_train_fold, y_train_fold)
    pred = model.predict_proba(X_test)
    predictions.append(pred)

# 평균 예측
final_prediction = np.mean(predictions, axis=0)
```

---

### 8. 스태킹 앙상블 개선 ⭐

**문제**: 현재 스태킹이 가중합보다 성능이 낮음

**해결책 A: 메타 모델 변경**
```python
# Logistic Regression 대신 다른 메타 모델 사용
from sklearn.ensemble import GradientBoostingClassifier

meta_model = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
```

**해결책 B: 다층 스태킹**
```python
# 첫 번째 레벨: Base models
# 두 번째 레벨: 첫 번째 레벨의 예측 + 원본 피처
meta_X = np.column_stack([
    oof_predictions['LogisticRegression'],
    oof_predictions['XGBoost'],
    oof_predictions['RandomForest'],
    X_train.iloc[:, :10]  # 원본 피처 일부 추가
])
```

---

## 우선순위별 실행 계획

### 즉시 실행 (1일 이내)
1. ✅ **임계값 최적화**: 0.5 → 0.35로 변경
2. ✅ **클래스 가중치 추가**: 모든 모델에 `class_weight='balanced'` 적용

### 단기 개선 (1주일 이내)
3. ✅ **하이퍼파라미터 튜닝**: XGBoost, RandomForest 정규화 강화
4. ✅ **앙상블 가중치 재조정**: Precision 기반 가중치 계산

### 중기 개선 (2주일 이내)
5. ✅ **피처 엔지니어링**: 상호작용 피처, 통계 피처 추가
6. ✅ **교차 검증 앙상블**: 다중 폴드 예측 평균

### 장기 개선 (1개월 이내)
7. ✅ **새로운 모델 추가**: CatBoost, Neural Network
8. ✅ **스태킹 개선**: 메타 모델 변경, 다층 스태킹

---

## 예상 성능 향상

| 개선 방안 | 예상 정확도 향상 | 예상 F1 향상 |
|----------|----------------|-------------|
| 임계값 최적화 | +0.5% | +0.6% |
| 클래스 불균형 처리 | +2-3% | +1-2% |
| 하이퍼파라미터 튜닝 | +1-2% | +0.5-1% |
| 앙상블 가중치 재조정 | +0.5-1% | +0.3-0.5% |
| 피처 엔지니어링 | +1-3% | +0.5-1.5% |
| **전체 합계** | **+5-10%** | **+3-6%** |

---

## 구현 체크리스트

- [ ] 임계값을 0.35로 변경하는 코드 작성
- [ ] 모든 모델에 class_weight='balanced' 추가
- [ ] XGBoost 하이퍼파라미터 튜닝 (정규화 강화)
- [ ] RandomForest 하이퍼파라미터 튜닝 (더 보수적)
- [ ] Precision 기반 앙상블 가중치 계산 함수 작성
- [ ] SMOTE 오버샘플링 실험
- [ ] 상호작용 피처 생성 코드 작성
- [ ] 교차 검증 앙상블 구현
- [ ] CatBoost 모델 추가 및 실험
- [ ] 개선된 모델로 재학습 및 submission 생성

---

## 참고사항

1. **현재 정확도가 40%인 것은 매우 낮은 수준**입니다. 이는 모델이 거의 랜덤 예측에 가까운 상태일 수 있음을 의미합니다.

2. **False Positive가 과도한 이유**:
   - 클래스 불균형 (1 클래스가 적을 가능성)
   - 모델이 보수적이지 않음
   - 피처의 예측력 부족

3. **개선 시 주의사항**:
   - 과적합 방지 (교차 검증 필수)
   - 테스트 세트 성능 모니터링
   - 단계별 개선 및 검증

