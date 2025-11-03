"""
앙상블 전략 구현
"""

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from utils.evaluation import macro_f1_score


class WeightedEnsemble:
    """가중 평균 앙상블"""
    
    def __init__(self, models, weights=None):
        """
        Parameters:
        -----------
        models : list
            학습된 모델 리스트
        weights : list, optional
            각 모델의 가중치 (None이면 균등 가중치)
        """
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights) / np.sum(weights)
    
    def predict_proba(self, X):
        """
        확률 예측 (가중 평균)
        
        Parameters:
        -----------
        X : np.ndarray
            입력 데이터
            
        Returns:
        --------
        y_proba : np.ndarray
            가중 평균된 확률
        """
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        # 가중 평균
        y_proba = np.average(predictions, axis=0, weights=self.weights)
        return y_proba
    
    def predict(self, X):
        """
        클래스 예측
        
        Parameters:
        -----------
        X : np.ndarray
            입력 데이터
            
        Returns:
        --------
        y_pred : np.ndarray
            예측된 클래스
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)


def optimize_ensemble_weights(models, X, y, cv_splits, n_trials=50):
    """
    앙상블 가중치 최적화 (Optuna 사용)
    
    Parameters:
    -----------
    models : list
        학습된 모델 리스트
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿
    n_trials : int
        Optuna trial 개수
        
    Returns:
    --------
    best_weights : np.ndarray
        최적 가중치
    best_score : float
        최적 점수
    """
    import optuna
    
    def objective(trial):
        # 가중치 제안
        weights = []
        for i in range(len(models)):
            weights.append(trial.suggest_float(f'weight_{i}', 0.0, 1.0))
        
        # 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        cv_scores = []
        
        for train_idx, val_idx in cv_splits:
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # 각 모델의 예측 확률
            predictions = []
            for model in models:
                # 모델이 CV split에 맞게 학습되어 있다고 가정
                # 실제로는 각 fold별로 다른 모델 인스턴스가 필요
                pred = model.predict_proba(X_val)
                predictions.append(pred)
            
            # 가중 평균
            y_proba = np.average(predictions, axis=0, weights=weights)
            y_pred = np.argmax(y_proba, axis=1)
            
            f1 = macro_f1_score(y_val, y_pred)
            cv_scores.append(f1)
            
            trial.report(f1, len(cv_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='ensemble_weights_optimization'
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_weights = []
    for i in range(len(models)):
        best_weights.append(study.best_params[f'weight_{i}'])
    
    best_weights = np.array(best_weights)
    best_weights = best_weights / best_weights.sum()
    
    print(f"\n앙상블 가중치 최적화 완료!")
    print(f"Best Macro-F1: {study.best_value:.6f}")
    print(f"Best Weights: {best_weights}")
    
    return best_weights, study.best_value


def create_voting_ensemble(models, voting='soft'):
    """
    Voting 앙상블 생성
    
    Parameters:
    -----------
    models : list of tuples
        (name, model) 튜플 리스트
    voting : str
        'soft' or 'hard'
        
    Returns:
    --------
    ensemble : VotingClassifier
        Voting 앙상블 모델
    """
    ensemble = VotingClassifier(
        estimators=models,
        voting=voting,
        n_jobs=-1
    )
    return ensemble


def create_stacking_ensemble(models, meta_model=None):
    """
    Stacking 앙상블 생성
    
    Parameters:
    -----------
    models : list of tuples
        (name, model) 튜플 리스트
    meta_model : object, optional
        메타 모델 (None이면 LogisticRegression 사용)
        
    Returns:
    --------
    ensemble : StackingClassifier
        Stacking 앙상블 모델
    """
    if meta_model is None:
        meta_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
    
    ensemble = StackingClassifier(
        estimators=models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    return ensemble

