# ============================================
# QDA 단일 모델 랜덤 서치 탐색
# - Macro-F1 최적화
# - Scaler 타입: None, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
# - reg_param: 0.0~5.0 균등 샘플링
# - tol: 1e-5, 1e-4, 1e-3 중 선택
# - 실행시간 최적화: CV=5, 샘플 수 조정 가능
# ============================================

import numpy as np
import pandas as pd
import warnings
import random
from scipy.stats import uniform
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

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

X_qda = X_full.drop(columns=to_drop)
X_test_qda = X_test_full.drop(columns=to_drop)

print(f"[Feature Selection] 제거된 피처 수: {len(to_drop)}")
print(f"[QDA용] {X_qda.shape[1]}개 피처 사용")

# ---------- CV helper ----------
def get_cv_splits(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(X, y):
        yield X.iloc[tr_idx], X.iloc[val_idx], y.iloc[tr_idx], y.iloc[val_idx]

# ---------- Transformer 생성 헬퍼 함수 ----------
def create_transformer(scaler_name, seed=42):
    """Scaler 이름에 따라 Transformer 인스턴스 생성"""
    if scaler_name == 'none':
        return None
    elif scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'robust':
        return RobustScaler()
    elif scaler_name == 'power_yeo':
        return PowerTransformer(method='yeo-johnson', standardize=True)
    elif scaler_name == 'power_boxcox':
        return PowerTransformer(method='box-cox', standardize=True)
    elif scaler_name == 'quantile_normal':
        return QuantileTransformer(output_distribution='normal', random_state=seed)
    elif scaler_name == 'quantile_uniform':
        return QuantileTransformer(output_distribution='uniform', random_state=seed)
    else:
        return None

# --- QDA 하이퍼파라미터 튜닝 (랜덤 서치, Macro-F1 기준) ---
def tune_qda_cv(X, y, n_samples=30, n_splits=5, seed=42):
    """
    QDA 랜덤 서치 튜닝
    - scaler_type: None, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
    - reg_param: 0.0~1.0 균등 샘플링 (QDA는 0.0~1.0 범위만 허용)
    - tol: 1e-5, 1e-4, 1e-3 중 선택
    - 실패 시 reg_param을 점진적으로 증가시키는 fallback 전략 적용 (최대 1.0까지)
    - PowerTransformer box-cox는 자동으로 yeo-johnson으로 대체 (음수 값 처리)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Transformer 타입 정의
    scaler_types = [
        ('none', None),
        ('standard', StandardScaler()),
        ('robust', RobustScaler()),
        ('power_yeo', PowerTransformer(method='yeo-johnson', standardize=True)),
        ('power_boxcox', PowerTransformer(method='box-cox', standardize=True)),
        ('quantile_normal', QuantileTransformer(output_distribution='normal', random_state=seed)),
        ('quantile_uniform', QuantileTransformer(output_distribution='uniform', random_state=seed)),
    ]
    
    tol_options = [1e-5, 1e-4, 1e-3]
    
    candidates = []
    print(f"\n{'='*60}")
    print(f"QDA 랜덤 서치 시작 (Macro-F1 기준, {n_samples}개 샘플, CV={n_splits})...")
    print(f"{'='*60}\n")
    
    for i in range(n_samples):
        # reg_param 샘플링 (QDA는 0.0~1.0 범위만 허용)
        # 작은 값에서 실패할 수 있으므로 0.01부터 시작
        base_reg_param = uniform(0.01, 1.0).rvs()
        
        # scaler_type 샘플링
        scaler_name, _ = random.choice(scaler_types)
        
        # tol 샘플링
        tol = random.choice(tol_options)
        
        # 샘플 전체에 대해 retry 전략 적용
        # reg_param을 점진적으로 증가시키는 전략 (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
        reg_param_candidates = [
            base_reg_param,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]
        
        max_sample_retries = len(reg_param_candidates)
        sample_success = False
        last_error_msg = None
        
        for sample_retry in range(max_sample_retries):
            # retry 시 reg_param 증가
            if sample_retry < len(reg_param_candidates):
                reg_param = reg_param_candidates[sample_retry]
            else:
                reg_param = 1.0  # 최대값
            
            f1s = []
            accs = []
            failed_folds = []
            error_messages = []
            
            # 모든 fold 시도
            for fold_idx, (X_tr, X_val, y_tr, y_val) in enumerate(get_cv_splits(X, y, n_splits=n_splits, seed=seed)):
                steps = [("imp", SimpleImputer(strategy="mean"))]
                
                # Transformer 추가
                transformer = create_transformer(scaler_name, seed=seed)
                
                # PowerTransformer box-cox는 양수 값만 가능하므로 사전 체크
                if scaler_name == 'power_boxcox':
                    X_tr_imp = SimpleImputer(strategy="mean").fit_transform(X_tr)
                    if (X_tr_imp <= 0).any().any():
                        # 음수 값이 있으면 yeo-johnson으로 대체
                        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                        scaler_name_actual = 'power_yeo'
                    else:
                        scaler_name_actual = scaler_name
                else:
                    scaler_name_actual = scaler_name
                
                if transformer is not None:
                    steps.append(("scaler", transformer))
                
                steps.append(("clf", QDA(reg_param=reg_param, tol=tol)))
                pipe = Pipeline(steps)
                
                try:
                    pipe.fit(X_tr, y_tr)
                    pred = pipe.predict(X_val)
                    f1m = f1_score(y_val, pred, average="macro")
                    acc = accuracy_score(y_val, pred)
                    f1s.append(f1m)
                    accs.append(acc)
                except Exception as e:
                    failed_folds.append(fold_idx)
                    error_msg = str(e)[:100]  # 에러 메시지 처음 100자만 저장
                    if error_msg not in error_messages:
                        error_messages.append(error_msg)
                    last_error_msg = error_msg
                    continue
            
            # 최소 3개 fold 이상 성공하면 유효한 결과로 간주
            min_success_folds = max(3, n_splits - 2)
            if len(f1s) >= min_success_folds:
                sample_success = True
                break
        
        # 샘플 전체가 실패한 경우
        if not sample_success:
            error_info = f"에러: {last_error_msg}" if last_error_msg else "알 수 없는 에러"
            print(f"[QDA {i+1}/{n_samples}] 실패 (모든 retry 실패, 마지막 성공 fold: {len(f1s)}/{n_splits}, {error_info})")
            continue
        
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_acc = np.mean(accs)
        
        candidates.append({
            'scaler_type': scaler_name,
            'reg_param': reg_param,
            'tol': tol,
            'f1': mean_f1,
            'f1_std': std_f1,
            'acc': mean_acc,
            'success_folds': len(f1s)
        })
        
        print(f"[QDA {i+1}/{n_samples}] scaler={scaler_name:20s}, reg={reg_param:.4f}, tol={tol:.0e} | "
              f"CV-MacroF1={mean_f1:.4f} (±{std_f1:.4f}), ACC={mean_acc:.4f} (성공 fold: {len(f1s)}/{n_splits})")
    
    if not candidates:
        print("\n[경고] 모든 QDA 샘플이 실패했습니다. 기본값 사용.")
        return {
            'scaler_type': 'standard',
            'reg_param': 0.0,
            'tol': 1e-4,
            'f1': 0.0
        }
    
    # 최고 성능 파라미터 선택
    best = max(candidates, key=lambda x: x['f1'])
    
    print(f"\n{'='*60}")
    print(f"[QDA] 최적 파라미터 발견!")
    print(f"{'='*60}")
    print(f"  Scaler Type: {best['scaler_type']}")
    print(f"  reg_param: {best['reg_param']:.4f}")
    print(f"  tol: {best['tol']:.0e}")
    print(f"  CV-MacroF1: {best['f1']:.4f} (±{best['f1_std']:.4f})")
    print(f"  CV-Accuracy: {best['acc']:.4f}")
    print(f"{'='*60}\n")
    
    # 상위 5개 결과 출력
    candidates_sorted = sorted(candidates, key=lambda x: x['f1'], reverse=True)
    print("상위 5개 결과:")
    for idx, cand in enumerate(candidates_sorted[:5], 1):
        print(f"  {idx}. scaler={cand['scaler_type']:20s}, reg={cand['reg_param']:.4f}, "
              f"tol={cand['tol']:.0e} | F1={cand['f1']:.4f}")
    print()
    
    return best

# =========================================================
# 메인 실행: reg_param 파라미터 탐색 (0~1 점진적 변화)
# =========================================================

# 고정 파라미터
FIXED_SCALER = StandardScaler()
FIXED_TOL = 1e-4

# 탐색할 reg_param 값 목록 (0~1 사이 점진적 변화)
REG_PARAM_VALUES = [
    0.0,
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.1, 0.15, 0.2, 0.25, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0
]

print("\n" + "="*60)
print("=== reg_param 파라미터 탐색 (0~1 점진적 변화) ===")
print(f"고정 파라미터: scaler=StandardScaler, tol={FIXED_TOL:.0e}")
print("="*60)

reg_results = []

for test_idx, reg_value in enumerate(REG_PARAM_VALUES, 1):
    print(f"\n{'='*60}")
    print(f"=== 테스트 {test_idx}/{len(REG_PARAM_VALUES)}: reg_param={reg_value:.3f} ===")
    print(f"{'='*60}")
    
    f1s_cv = []
    accs_cv = []
    
    # CV 성능 검증
    for fold_idx, (X_tr, X_val, y_tr, y_val) in enumerate(get_cv_splits(X_qda, y, n_splits=5, seed=42), 1):
        steps = [
            ("imp", SimpleImputer(strategy="mean")),
            ("scaler", FIXED_SCALER),
            ("clf", QDA(reg_param=reg_value, tol=FIXED_TOL))
        ]
        pipe = Pipeline(steps)
        
        try:
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)
            f1m = f1_score(y_val, pred, average="macro")
            acc = accuracy_score(y_val, pred)
            f1s_cv.append(f1m)
            accs_cv.append(acc)
            print(f"[Fold {fold_idx}] Macro-F1: {f1m:.4f}, Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"[Fold {fold_idx}] 실패: {str(e)[:50]}")
            continue
    
    if len(f1s_cv) < 3:
        print(f"\n⚠️  reg_param={reg_value:.3f}: 실패한 fold가 너무 많아 결과에서 제외")
        continue
    
    mean_f1_cv = np.mean(f1s_cv)
    std_f1_cv = np.std(f1s_cv)
    mean_acc_cv = np.mean(accs_cv)
    
    reg_results.append({
        'reg_param': reg_value,
        'tol': FIXED_TOL,
        'scaler_name': 'standard',
        'cv_f1': mean_f1_cv,
        'cv_f1_std': std_f1_cv,
        'cv_acc': mean_acc_cv,
        'success_folds': len(f1s_cv)
    })
    
    print(f"\n[reg_param={reg_value:.3f} 결과]")
    print(f"  CV-Macro-F1: {mean_f1_cv:.4f} (±{std_f1_cv:.4f})")
    print(f"  CV-Accuracy: {mean_acc_cv:.4f}")
    print(f"  성공 fold: {len(f1s_cv)}/5")

# 결과 요약
print(f"\n{'='*60}")
print("=== reg_param 파라미터 비교 결과 요약 ===")
print(f"{'='*60}")

# 성능 순으로 정렬
reg_results_sorted = sorted(reg_results, key=lambda x: x['cv_f1'], reverse=True)

print(f"\n{'순위':<5} {'reg_param':<15} {'CV-Macro-F1':<20} {'CV-Accuracy':<15}")
print("-" * 70)
for idx, result in enumerate(reg_results_sorted, 1):
    print(f"{idx:<5} {result['reg_param']:.3f}  {result['cv_f1']:.4f} (±{result['cv_f1_std']:.4f})  {result['cv_acc']:.4f}")

# reg_param 값 순으로도 출력 (성능 변화 추이 확인)
print(f"\n{'='*60}")
print("=== reg_param 값 순서대로 성능 변화 ===")
print(f"{'='*60}")
print(f"\n{'reg_param':<15} {'CV-Macro-F1':<20} {'CV-Accuracy':<15}")
print("-" * 70)
for result in sorted(reg_results, key=lambda x: x['reg_param']):
    print(f"{result['reg_param']:.3f}  {result['cv_f1']:.4f} (±{result['cv_f1_std']:.4f})  {result['cv_acc']:.4f}")

# 최고 성능 reg_param 선택
best_reg_result = reg_results_sorted[0]
best_params = {
    'scaler_type': best_reg_result['scaler_name'],
    'reg_param': best_reg_result['reg_param'],
    'tol': best_reg_result['tol'],
    'f1': best_reg_result['cv_f1']
}

print(f"\n{'='*60}")
print("=== 최고 성능 reg_param 파라미터 ===")
print(f"{'='*60}")
print(f"  Scaler: {best_params['scaler_type']}")
print(f"  reg_param: {best_params['reg_param']:.4f}")
print(f"  tol: {best_params['tol']:.0e}")
print(f"  CV-Macro-F1: {best_params['f1']:.4f} (±{best_reg_result['cv_f1_std']:.4f})")
print(f"{'='*60}")

print("\n" + "="*60)
print("=== Step 2: 최적 파라미터로 최종 모델 학습 ===")
print("="*60)

# 최종 파이프라인 구성
steps_final = [("imp", SimpleImputer(strategy="mean"))]

transformer_final = create_transformer(best_params['scaler_type'], seed=42)
if transformer_final is not None:
    steps_final.append(("scaler", transformer_final))

steps_final.append(("clf", QDA(
    reg_param=best_params['reg_param'],
    tol=best_params['tol']
)))

pipe_qda_final = Pipeline(steps_final)
pipe_qda_final.fit(X_qda, y)

print("✓ QDA Final Model 학습 완료")
print(f"  사용된 Scaler: {best_params['scaler_type']}")
print(f"  reg_param: {best_params['reg_param']:.4f}")
print(f"  tol: {best_params['tol']:.0e}")

# 학습 데이터 성능 평가
train_pred = pipe_qda_final.predict(X_qda)
train_f1 = f1_score(y, train_pred, average="macro")
train_acc = accuracy_score(y, train_pred)
print(f"\n[학습 데이터 성능]")
print(f"  Macro-F1: {train_f1:.4f}")
print(f"  Accuracy: {train_acc:.4f}")

# =========================================================
# Step 3: 테스트 데이터 예측
# =========================================================

print("\n" + "="*60)
print("=== Step 3: 테스트 데이터 예측 ===")
print("="*60)

test_pred = pipe_qda_final.predict(X_test_qda)

out_path = "submission_qda_single_model.csv"
pd.DataFrame({"ID": test_ids, "target": test_pred}).to_csv(out_path, index=False)
print(f"\n[저장 완료] {out_path}")
print(f"[예상 성능] Macro-F1: {best_params['f1']:.4f} (CV 기준)")

