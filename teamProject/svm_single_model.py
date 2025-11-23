# ============================================
# SVM 단일 모델 성능 평가
# - Support Vector Machine
# - 가장 많이 사용되는 하이퍼파라미터 사용
# - Macro-F1 평가
# ============================================

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------- 데이터 로드 ----------
def load_data():
    try:
        df_train = pd.read_csv("train.csv")
        df_test  = pd.read_csv("test.csv")
        return df_train, df_test
    except FileNotFoundError:
        print("경고: train.csv/test.csv를 찾을 수 없습니다.")
        raise

train_df, test_df = load_data()
X_full = train_df.drop(['ID', 'target'], axis=1)
y = train_df['target'].astype(int)
test_ids = test_df['ID']
X_test_full = test_df.drop(['ID'], axis=1)
num_classes = len(np.unique(y))

print(f"Train: {X_full.shape[0]} | Feat: {X_full.shape[1]} | Test: {X_test_full.shape[0]} | Classes: {num_classes}")

# --- 강한 상관 피처 제거 (0.999 초과) ---
corr = X_full.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]

X_svm = X_full.drop(columns=to_drop)
X_test_svm = X_test_full.drop(columns=to_drop)

print(f"[Feature Selection] 제거된 피처 수: {len(to_drop)}")
print(f"[SVM용] {X_svm.shape[1]}개 피처 사용")

# ---------- CV helper ----------
def get_cv_splits(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(X, y):
        yield X.iloc[tr_idx], X.iloc[val_idx], y.iloc[tr_idx], y.iloc[val_idx]

# =========================================================
# CV 성능 검증
# =========================================================

print("\n" + "="*60)
print("=== SVM 모델 CV 성능 검증 ===")
print("="*60)

# 가장 많이 사용되는 SVM 하이퍼파라미터
# - C=1.0: 기본값, 정규화 강도
# - kernel='rbf': 가장 많이 사용되는 커널
# - gamma='scale': 기본값, 자동 스케일링
# - probability=True: 확률 예측 가능
# - random_state: 재현성 보장

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'random_state': 42
}

print(f"\n사용된 하이퍼파라미터:")
print(f"  C: {SVM_PARAMS['C']}")
print(f"  kernel: {SVM_PARAMS['kernel']}")
print(f"  gamma: {SVM_PARAMS['gamma']}")
print(f"  probability: {SVM_PARAMS['probability']}")

f1s_cv = []
accs_cv = []

for fold_idx, (X_tr, X_val, y_tr, y_val) in enumerate(get_cv_splits(X_svm, y, n_splits=5, seed=42), 1):
    # SVM 파이프라인 구성
    # SVM은 스케일링이 매우 중요하므로 StandardScaler 필수
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", SVC(**SVM_PARAMS))
    ])
    
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_val)
    f1m = f1_score(y_val, pred, average="macro")
    acc = accuracy_score(y_val, pred)
    f1s_cv.append(f1m)
    accs_cv.append(acc)
    print(f"[Fold {fold_idx}] Macro-F1: {f1m:.4f}, Accuracy: {acc:.4f}")

mean_f1_cv = np.mean(f1s_cv)
std_f1_cv = np.std(f1s_cv)
mean_acc_cv = np.mean(accs_cv)

print(f"\n[CV 결과]")
print(f"  CV-Macro-F1: {mean_f1_cv:.4f} (±{std_f1_cv:.4f})")
print(f"  CV-Accuracy: {mean_acc_cv:.4f}")

# =========================================================
# 최종 모델 학습
# =========================================================

print("\n" + "="*60)
print("=== 최종 SVM 모델 학습 ===")
print("="*60)

# 최종 파이프라인 구성
pipe_svm_final = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", SVC(**SVM_PARAMS))
])

pipe_svm_final.fit(X_svm, y)

print("✓ SVM Final Model 학습 완료")
print(f"  사용된 파라미터:")
print(f"    C: {SVM_PARAMS['C']}")
print(f"    kernel: {SVM_PARAMS['kernel']}")
print(f"    gamma: {SVM_PARAMS['gamma']}")
print(f"    scaler: StandardScaler")

# 학습 데이터 성능 평가
train_pred = pipe_svm_final.predict(X_svm)
train_f1 = f1_score(y, train_pred, average="macro")
train_acc = accuracy_score(y, train_pred)
print(f"\n[학습 데이터 성능]")
print(f"  Macro-F1: {train_f1:.4f}")
print(f"  Accuracy: {train_acc:.4f}")

# =========================================================
# 테스트 데이터 예측
# =========================================================

print("\n" + "="*60)
print("=== 테스트 데이터 예측 ===")
print("="*60)

test_pred = pipe_svm_final.predict(X_test_svm)

out_path = "submission_svm_single_model.csv"
pd.DataFrame({"ID": test_ids, "target": test_pred}).to_csv(out_path, index=False)
print(f"\n[저장 완료] {out_path}")
print(f"[예상 성능] Macro-F1: {mean_f1_cv:.4f} (CV 기준)")

