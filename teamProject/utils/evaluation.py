"""
평가 메트릭 모듈
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import pandas as pd


def macro_f1_score(y_true, y_pred):
    """
    Macro-F1 Score 계산
    
    Parameters:
    -----------
    y_true : np.ndarray
        실제 레이블
    y_pred : np.ndarray
        예측 레이블
        
    Returns:
    --------
    macro_f1 : float
        Macro-F1 Score
    """
    return f1_score(y_true, y_pred, average='macro')


def evaluate_model(y_true, y_pred, y_proba=None):
    """
    모델 평가 지표 계산
    
    Parameters:
    -----------
    y_true : np.ndarray
        실제 레이블
    y_pred : np.ndarray
        예측 레이블
    y_proba : np.ndarray, optional
        예측 확률
        
    Returns:
    --------
    metrics : dict
        평가 지표 딕셔너리
    """
    metrics = {
        'macro_f1': macro_f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'per_class_f1': f1_score(y_true, y_pred, average=None)
    }
    
    return metrics


def print_evaluation_report(y_true, y_pred, class_names=None):
    """
    평가 리포트 출력
    
    Parameters:
    -----------
    y_true : np.ndarray
        실제 레이블
    y_pred : np.ndarray
        예측 레이블
    class_names : list, optional
        클래스 이름 리스트
    """
    print("=" * 60)
    print("모델 평가 리포트")
    print("=" * 60)
    
    macro_f1 = macro_f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nMacro-F1 Score: {macro_f1:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))
    
    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    print(cm_df)
    
    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }


def cross_validate_model(model, X, y, cv_splits, verbose=True):
    """
    교차 검증 수행
    
    Parameters:
    -----------
    model : object
        학습 가능한 모델 객체 (fit, predict 메서드 필요)
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿 (train_idx, val_idx 튜플)
    verbose : bool
        출력 여부
        
    Returns:
    --------
    results : dict
        교차 검증 결과
    """
    cv_scores = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_val)
        
        # 평가
        macro_f1 = macro_f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        
        cv_scores.append(macro_f1)
        fold_results.append({
            'fold': fold + 1,
            'macro_f1': macro_f1,
            'accuracy': accuracy
        })
        
        if verbose:
            print(f"Fold {fold + 1}: Macro-F1 = {macro_f1:.6f}, Accuracy = {accuracy:.6f}")
    
    mean_f1 = np.mean(cv_scores)
    std_f1 = np.std(cv_scores)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"CV Results: {mean_f1:.6f} (+/- {std_f1:.6f})")
        print("=" * 60)
    
    results = {
        'mean_macro_f1': mean_f1,
        'std_macro_f1': std_f1,
        'scores': cv_scores,
        'fold_results': fold_results
    }
    
    return results

