"""
트리 기반 모델 구현 및 Optuna 최적화
"""

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import macro_f1_score


class LightGBMWrapper:
    """LightGBM 모델 래퍼"""
    
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
        
    def fit(self, X, y, **kwargs):
        """모델 학습"""
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params,
            train_data,
            **kwargs
        )
        return self
    
    def predict(self, X):
        """클래스 예측"""
        y_proba = self.model.predict(X)
        return np.argmax(y_proba.reshape(-1, 21), axis=1)
    
    def predict_proba(self, X):
        """확률 예측"""
        y_proba = self.model.predict(X)
        return y_proba.reshape(-1, 21)


class XGBoostWrapper:
    """XGBoost 모델 래퍼"""
    
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
        
    def fit(self, X, y, **kwargs):
        """모델 학습"""
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.params,
            dtrain,
            **kwargs
        )
        return self
    
    def predict(self, X):
        """클래스 예측"""
        dtest = xgb.DMatrix(X)
        y_proba = self.model.predict(dtest)
        return np.argmax(y_proba.reshape(-1, 21), axis=1)
    
    def predict_proba(self, X):
        """확률 예측"""
        dtest = xgb.DMatrix(X)
        y_proba = self.model.predict(dtest)
        return y_proba.reshape(-1, 21)


class CatBoostWrapper:
    """CatBoost 모델 래퍼"""
    
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
        
    def fit(self, X, y, **kwargs):
        """모델 학습"""
        self.model = cb.CatBoostClassifier(**self.params, **kwargs, verbose=False)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """클래스 예측"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """확률 예측"""
        return self.model.predict_proba(X)


def optimize_lightgbm(X, y, cv_splits, n_trials=100, timeout=None, n_jobs=1):
    """
    LightGBM 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿
    n_trials : int
        Optuna trial 개수
    timeout : float, optional
        최적화 시간 제한 (초)
    n_jobs : int
        병렬 작업 개수
        
    Returns:
    --------
    best_params : dict
        최적 하이퍼파라미터
    study : optuna.Study
        Optuna study 객체
    """
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 21,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            y_pred = np.argmax(model.predict(X_val).reshape(-1, 21), axis=1)
            f1 = macro_f1_score(y_val, y_pred)
            cv_scores.append(f1)
            
            # 중간 보고 (Pruner를 위해)
            trial.report(f1, len(cv_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name='lightgbm_optimization'
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'multiclass',
        'num_class': 21,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1
    })
    
    print(f"\nLightGBM 최적화 완료!")
    print(f"Best Macro-F1: {study.best_value:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, study


def optimize_xgboost(X, y, cv_splits, n_trials=100, timeout=None, n_jobs=1):
    """
    XGBoost 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿
    n_trials : int
        Optuna trial 개수
    timeout : float, optional
        최적화 시간 제한 (초)
    n_jobs : int
        병렬 작업 개수
        
    Returns:
    --------
    best_params : dict
        최적 하이퍼파라미터
    study : optuna.Study
        Optuna study 객체
    """
    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 21,
            'eval_metric': 'mlogloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            y_pred = np.argmax(model.predict(dval).reshape(-1, 21), axis=1)
            f1 = macro_f1_score(y_val, y_pred)
            cv_scores.append(f1)
            
            trial.report(f1, len(cv_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name='xgboost_optimization'
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 21,
        'eval_metric': 'mlogloss',
        'random_state': 42
    })
    
    print(f"\nXGBoost 최적화 완료!")
    print(f"Best Macro-F1: {study.best_value:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, study


def optimize_catboost(X, y, cv_splits, n_trials=100, timeout=None, n_jobs=1):
    """
    CatBoost 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿
    n_trials : int
        Optuna trial 개수
    timeout : float, optional
        최적화 시간 제한 (초)
    n_jobs : int
        병렬 작업 개수
        
    Returns:
    --------
    best_params : dict
        최적 하이퍼파라미터
    study : optuna.Study
        Optuna study 객체
    """
    def objective(trial):
        params = {
            'objective': 'MultiClass',
            'loss_function': 'MultiClass',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'random_state': 42,
            'verbose': False
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = cb.CatBoostClassifier(**params, iterations=1000)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            f1 = macro_f1_score(y_val, y_pred)
            cv_scores.append(f1)
            
            trial.report(f1, len(cv_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name='catboost_optimization'
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'MultiClass',
        'loss_function': 'MultiClass',
        'random_state': 42,
        'verbose': False
    })
    
    print(f"\nCatBoost 최적화 완료!")
    print(f"Best Macro-F1: {study.best_value:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, study

