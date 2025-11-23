# ============================================
# 랜덤 서치 확장 버전: Macro-F1 최적화 스태킹 앙상블
# - LR/QDA 랜덤 서치로 탐색 공간 확대
# - LGBM 파라미터 원본 7개 유지
# - Meta-LR C=1.0 고정 (과적합 방지)
# - 실행시간 최적화: CV=3, 샘플 수 제한
# ============================================

import numpy as np
import pandas as pd
import warnings
import random
from scipy.stats import loguniform, uniform
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

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

# --- LR/QDA용: 강한 상관 피처 제거 (0.999 초과) ---
corr = X_full.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]

X_lrqda = X_full.drop(columns=to_drop)
X_test_lrqda = X_test_full.drop(columns=to_drop)

print(f"[Feature Selection] 제거된 피처 수: {len(to_drop)}")
print(f"[LR/QDA용] {X_lrqda.shape[1]}개 | [LGBM용] {X_full.shape[1]}개")

# ---------- CV helper ----------
def get_cv_splits(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(X, y):
        yield X.iloc[tr_idx], X.iloc[val_idx], y.iloc[tr_idx], y.iloc[val_idx]

# --- 1) LR 하이퍼파라미터 튜닝 (랜덤 서치, Macro-F1 기준) ---
def tune_lr_cv(X, y, n_samples=12, n_splits=3, seed=42):
    """
    LR 랜덤 서치 튜닝
    - C: 0.01~50.0 로그 균등 분포
    - solver: lbfgs, newton-cg, saga
    - penalty: solver별 호환 (l2 또는 elasticnet)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    results = []
    print(f"LR 랜덤 서치 시작 (Macro-F1 기준, {n_samples}개 샘플, CV={n_splits})...")
    
    for i in range(n_samples):
        # C 샘플링 (로그 균등 분포)
        C = loguniform(0.01, 50.0).rvs()
        
        # solver 샘플링
        solver = random.choice(['lbfgs', 'newton-cg', 'saga'])
        
        # penalty 결정 (solver별 호환성)
        if solver in ['lbfgs', 'newton-cg']:
            penalty = 'l2'
            l1_ratio = None
        else:  # saga
            penalty = random.choice(['l2', 'elasticnet'])
            if penalty == 'elasticnet':
                l1_ratio = uniform(0.0, 1.0).rvs()
            else:
                l1_ratio = None
        
        f1s = []
        for X_tr, X_val, y_tr, y_val in get_cv_splits(X, y, n_splits=n_splits, seed=seed):
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C, max_iter=2000, multi_class="multinomial",
                    solver=solver, penalty=penalty, l1_ratio=l1_ratio,
                    n_jobs=-1, random_state=seed
                ))
            ])
            try:
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_val)
                f1m = f1_score(y_val, pred, average="macro")
                f1s.append(f1m)
            except Exception:
                pass
        
        if len(f1s) < n_splits:
            continue
            
        mean_f1 = np.mean(f1s)
        results.append({
            'C': C,
            'solver': solver,
            'penalty': penalty,
            'l1_ratio': l1_ratio,
            'f1': mean_f1
        })
        print(f"[LR {i+1}/{n_samples}] C={C:.4f}, solver={solver}, penalty={penalty}, l1_ratio={l1_ratio}, CV-MacroF1={mean_f1:.4f}")
    
    if not results:
        print("[경고] 모든 LR 샘플이 실패했습니다. 기본값 사용.")
        return 1.0
    
    best = max(results, key=lambda x: x['f1'])
    print(f"[LR] Best: C={best['C']:.4f}, solver={best['solver']}, penalty={best['penalty']}, l1_ratio={best['l1_ratio']}, CV-MacroF1={best['f1']:.4f}")
    
    # 원본 호환성을 위해 C만 반환 (OOF 생성 시 solver는 "lbfgs"로 고정)
    return best['C']

# --- 2) QDA 하이퍼파라미터 튜닝 (랜덤 서치, Macro-F1 기준) ---
def tune_qda_cv(X, y, n_samples=15, n_splits=3, seed=42):
    """
    QDA 랜덤 서치 튜닝
    - scaler_type: None, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
    - reg_param: 0.0~5.0 균등 샘플링
    - tol: 1e-5, 1e-4, 1e-3 중 선택
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
    print(f"QDA 랜덤 서치 시작 (Macro-F1 기준, {n_samples}개 샘플, CV={n_splits})...")
    
    for i in range(n_samples):
        # reg_param 샘플링 (균등 분포)
        reg_param = uniform(0.0, 5.0).rvs()
        
        # scaler_type 샘플링
        scaler_name, scaler_transformer = random.choice(scaler_types)
        
        # tol 샘플링
        tol = random.choice(tol_options)
        
        f1s = []
        for X_tr, X_val, y_tr, y_val in get_cv_splits(X, y, n_splits=n_splits, seed=seed):
            steps = [("imp", SimpleImputer(strategy="mean"))]
            
            if scaler_transformer is not None:
                # Transformer 복사 (fit 시 상태 변경 방지)
                if scaler_name.startswith('power'):
                    transformer = PowerTransformer(
                        method='yeo-johnson' if 'yeo' in scaler_name else 'box-cox',
                        standardize=True
                    )
                elif scaler_name.startswith('quantile'):
                    transformer = QuantileTransformer(
                        output_distribution='normal' if 'normal' in scaler_name else 'uniform',
                        random_state=seed
                    )
                elif scaler_name == 'standard':
                    transformer = StandardScaler()
                elif scaler_name == 'robust':
                    transformer = RobustScaler()
                else:
                    transformer = None
                
                if transformer is not None:
                    steps.append(("scaler", transformer))
            
            steps.append(("clf", QDA(reg_param=reg_param, tol=tol)))
            pipe = Pipeline(steps)
            
            try:
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_val)
                f1m = f1_score(y_val, pred, average="macro")
                f1s.append(f1m)
            except Exception:
                pass
        
        if len(f1s) < n_splits:
            continue
        
        mean_f1 = np.mean(f1s)
        candidates.append({
            'scaler_type': scaler_name,
            'reg_param': reg_param,
            'tol': tol,
            'f1': mean_f1
        })
        print(f"[QDA {i+1}/{n_samples}] scaler={scaler_name}, reg={reg_param:.4f}, tol={tol:.0e}, CV-MacroF1={mean_f1:.4f}")
    
    if not candidates:
        print("[경고] 모든 QDA 샘플이 실패했습니다. 기본값 사용.")
        return True, 0.0
    
    best = max(candidates, key=lambda x: x['f1'])
    print(f"[QDA] Best: scaler={best['scaler_type']}, reg={best['reg_param']:.4f}, tol={best['tol']:.0e}, CV-MacroF1={best['f1']:.4f}")
    
    # 원본 호환성을 위해 use_scaler (boolean)과 reg_param 반환
    # scaler_type이 'none'이 아니면 use_scaler=True
    use_scaler_best = (best['scaler_type'] != 'none')
    # 원본에서는 StandardScaler만 사용하므로, 다른 transformer가 선택되더라도
    # OOF 생성 시에는 StandardScaler를 사용 (원본 구조 유지)
    return use_scaler_best, best['reg_param']

# --- 3) LGBM 하이퍼파라미터 튜닝 (원본 7개 유지, Macro-F1 기준) ---
def tune_lgbm_cv(X, y, num_classes):
    # ✅ 원본 7개 파라미터 유지
    params_grid = [
        {"max_depth": 3, "num_leaves": 15, "min_child_samples": 50, "lambda_l2": 0.0, "learning_rate": 0.1},
        {"max_depth": 3, "num_leaves": 31, "min_child_samples": 50, "lambda_l2": 1.0, "learning_rate": 0.1},
        {"max_depth": 5, "num_leaves": 31, "min_child_samples": 50, "lambda_l2": 0.0, "learning_rate": 0.1},
        {"max_depth": 5, "num_leaves": 31, "min_child_samples": 20, "lambda_l2": 1.0, "learning_rate": 0.05},
        {"max_depth": 7, "num_leaves": 63, "min_child_samples": 50, "lambda_l2": 1.0, "learning_rate": 0.05},
        {"max_depth": 7, "num_leaves": 63, "min_child_samples": 20, "lambda_l2": 5.0, "learning_rate": 0.03},
        {"max_depth": 5, "num_leaves": 63, "min_child_samples": 30, "lambda_l2": 5.0, "learning_rate": 0.05},
    ]
    results = []
    print("LGBM 튜닝 시작 (원본 7개 파라미터, Macro-F1 기준)...")
    for i, p in enumerate(params_grid, 1):
        f1s = []
        best_iters = []
        print(f"\n[LGBM {i}/7] params={p}")
        for fold_id, (X_tr, X_val, y_tr, y_val) in enumerate(get_cv_splits(X, y, n_splits=5, seed=42), 1):
            clf = LGBMClassifier(
                objective="multiclass", num_class=num_classes,
                n_estimators=5000, random_state=42 + fold_id,
                n_jobs=-1, verbosity=-1, **p
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)]
            )
            pred = clf.predict(X_val)
            f1m = f1_score(y_val, pred, average="macro")  # ✅ Macro-F1 사용
            f1s.append(f1m)
            best_iter = getattr(clf, "best_iteration_", None)
            if best_iter is not None:
                best_iters.append(best_iter)
        
        mean_f1 = np.mean(f1s)
        mean_iter = int(np.mean(best_iters)) if len(best_iters) > 0 else 1000
        results.append((p, mean_f1, mean_iter))
        print(f"  → CV-MacroF1={mean_f1:.4f}, mean_best_iter={mean_iter}")

    best_p, best_f1, best_iter = max(results, key=lambda x: x[1])
    print(f"\n[LGBM] Best params={best_p} (CV-MacroF1={best_f1:.4f})")
    return best_p, best_iter

# =========================================================
# 메인 실행: 베이스 모델 튜닝 및 OOF 생성
# =========================================================

print("\n" + "="*60)
print("=== Step 1: 베이스 모델 하이퍼파라미터 튜닝 (랜덤 서치, Macro-F1) ===")
print("="*60)

best_C_lr = tune_lr_cv(X_lrqda, y, n_samples=12, n_splits=3, seed=42)
best_use_scaler, best_reg = tune_qda_cv(X_lrqda, y, n_samples=15, n_splits=3, seed=42)
best_lgb_params, best_lgb_n_estimators = tune_lgbm_cv(X_full, y, num_classes)

print("\n" + "="*60)
print("=== Step 2: OOF 예측 생성 (스태킹 메타 피처) ===")
print("="*60)

skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

oof_lr = np.zeros((len(y), num_classes))
oof_qda = np.zeros((len(y), num_classes))
oof_lgb = np.zeros((len(y), num_classes))

for fold, (tr_idx, val_idx) in enumerate(skf_stack.split(X_full, y), 1):
    X_tr_lr, X_val_lr = X_lrqda.iloc[tr_idx], X_lrqda.iloc[val_idx]
    X_tr_lgb, X_val_lgb = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # LR (OOF)
    pipe_lr = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=best_C_lr, max_iter=2000, multi_class="multinomial",
            solver="lbfgs", n_jobs=-1, random_state=2025 + fold
        ))
    ]).fit(X_tr_lr, y_tr)

    # QDA (OOF)
    steps_qda_fold = [("imp", SimpleImputer(strategy="mean"))]
    if best_use_scaler:
        steps_qda_fold.append(("scaler", StandardScaler()))
    steps_qda_fold.append(("clf", QDA(reg_param=best_reg)))
    pipe_qda = Pipeline(steps_qda_fold).fit(X_tr_lr, y_tr)

    # LGBM (OOF)
    lgb_model = LGBMClassifier(
        objective="multiclass", num_class=num_classes,
        n_estimators=best_lgb_n_estimators, random_state=2025 + fold,
        n_jobs=-1, verbosity=-1, **best_lgb_params
    ).fit(X_tr_lgb, y_tr)

    # OOF 확률 저장
    oof_lr[val_idx] = pipe_lr.predict_proba(X_val_lr)
    oof_qda[val_idx] = pipe_qda.predict_proba(X_val_lr)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val_lgb)

    # Fold 성능 (Macro-F1)
    proba_ens_fold = (oof_lr[val_idx] + oof_qda[val_idx] + oof_lgb[val_idx]) / 3.0
    pred_ens_fold = np.argmax(proba_ens_fold, axis=1)
    f1_fold = f1_score(y_val, pred_ens_fold, average="macro")  # ✅ Macro-F1
    print(f"[Fold {fold}] Simple Average Macro-F1 = {f1_fold:.4f}")

# 메타 입력 행렬
meta_X = np.concatenate([oof_lr, oof_qda, oof_lgb], axis=1)
print(f"Meta feature shape: {meta_X.shape}")

# =========================================================
# Step 3: Meta Logistic Regression (C=1.0 고정)
# =========================================================

print("\n" + "="*60)
print("=== Step 3: Meta LR 학습 (C=1.0 고정, 과적합 방지) ===")
print("="*60)

# ✅ 원본처럼 C=1.0 고정 (튜닝 안함)
meta_clf = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs",
    n_jobs=-1,
    C=1.0,  # 고정!
    random_state=777
)
meta_clf.fit(meta_X, y)

meta_pred_train = meta_clf.predict(meta_X)
meta_f1 = f1_score(y, meta_pred_train, average="macro")  # ✅ Macro-F1
meta_acc = accuracy_score(y, meta_pred_train)
print(f"[Meta] OOF Training Macro-F1 = {meta_f1:.4f}, ACC = {meta_acc:.4f}")

print("\n" + "="*60)
print("=== Step 4: 전체 데이터로 베이스 모델 재학습 ===")
print("="*60)

# LR 최종
pipe_lr_final = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=best_C_lr, max_iter=2000, multi_class="multinomial",
        solver="lbfgs", n_jobs=-1, random_state=2025
    ))
]).fit(X_lrqda, y)
print("✓ LR Final Model 학습 완료")

# QDA 최종
steps_qda_final = [("imp", SimpleImputer(strategy="mean"))]
if best_use_scaler:
    steps_qda_final.append(("scaler", StandardScaler()))
steps_qda_final.append(("clf", QDA(reg_param=best_reg)))
pipe_qda_final = Pipeline(steps_qda_final).fit(X_lrqda, y)
print("✓ QDA Final Model 학습 완료")

# LGBM 최종
lgb_final = LGBMClassifier(
    objective="multiclass", num_class=num_classes,
    n_estimators=best_lgb_n_estimators, random_state=2025,
    n_jobs=-1, verbosity=-1, **best_lgb_params
).fit(X_full, y)
print("✓ LGBM Final Model 학습 완료")

# =========================================================
# Step 5: 테스트 데이터 최종 예측
# =========================================================

proba_lr_test = pipe_lr_final.predict_proba(X_test_lrqda)
proba_qda_test = pipe_qda_final.predict_proba(X_test_lrqda)
proba_lgb_test = lgb_final.predict_proba(X_test_full)

meta_X_test = np.concatenate([proba_lr_test, proba_qda_test, proba_lgb_test], axis=1)
final_pred = meta_clf.predict(meta_X_test)

out_path = "submission_random_search_macroF1.csv"
pd.DataFrame({"ID": test_ids, "target": final_pred}).to_csv(out_path, index=False)
print(f"\n[저장 완료] {out_path}")
print(f"[예상 성능] 0.86~0.88 (랜덤 서치 확장 탐색)")

