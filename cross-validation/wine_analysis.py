#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wine 데이터셋 교차검증 및 통계 검정 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
try:
    # statannot는 수평 브래킷과 별표 표기를 위해 사용
    from statannot import add_stat_annotation
    _HAS_STATANNOT = True
except Exception:
    _HAS_STATANNOT = False

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """Wine 데이터 로드 및 전처리"""
    print("Wine 데이터 로드 중...")
    df = pd.read_csv("../wine-classification-dacon/data/train.csv", index_col=None)
    
    print(f"데이터셋 크기: {df.shape}")
    print(f"클래스 분포:")
    print(df['quality'].value_counts().sort_index())
    
    # 특성과 타겟 분리 (문자열 컬럼 제외)
    X = df.drop(['index', 'quality', 'type'], axis=1)
    y = df['quality']
    
    print(f"\n특성 수: {X.shape[1]}")
    print(f"샘플 수: {X.shape[0]}")
    print(f"클래스 수: {len(y.unique())}")
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n데이터 표준화 완료")
    print(f"표준화 전 평균: {X.mean().mean():.4f}")
    print(f"표준화 후 평균: {X_scaled.mean():.4f}")
    print(f"표준화 후 표준편차: {X_scaled.std():.4f}")
    
    return X_scaled, y

def get_classifiers():
    """8개 분류 알고리즘 정의"""
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'GaussianNB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'NeuralNetwork': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    print("사용할 분류 알고리즘:")
    for i, (name, _) in enumerate(classifiers.items(), 1):
        print(f"{i}. {name}")
    
    return classifiers

def perform_5fold_cv(X, y, classifiers):
    """Stratified 5-fold Cross Validation 수행"""
    print("\nStratified 5-fold Cross Validation 수행 중...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, classifier in classifiers.items():
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        
        cv_results[name] = scores
        print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print("5-fold CV 완료!")
    return cv_results

def perform_shuffle_split_cv(X, y, classifiers):
    """Stratified Shuffle Split Cross Validation (40회) 수행"""
    print("\nStratified Shuffle Split Cross Validation (40회) 수행 중...")
    
    sss = StratifiedShuffleSplit(n_splits=40, test_size=0.5, random_state=42)
    cv_results = {}
    
    for name, classifier in classifiers.items():
        scores = []
        for train_idx, test_idx in sss.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        
        cv_results[name] = scores
        print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print("Shuffle Split CV 완료!")
    return cv_results

def perform_statistical_tests(results_df, best_model_name):
    """최고 성능 모델과 다른 모델들 간의 통계 검정 수행"""
    best_scores = results_df[best_model_name]
    p_values = {}
    
    for model in results_df.columns:
        if model != best_model_name:
            other_scores = results_df[model]
            # Wilcoxon signed-rank test (paired test)
            statistic, p_value = stats.wilcoxon(best_scores, other_scores, alternative='greater')
            p_values[model] = p_value
    
    return p_values

def create_barplot_with_significance(results_df, p_values, best_model, title, filename):
    """유의성을 *로 표기한 bar plot 생성 (statannot 스타일)"""
    means = results_df.mean()
    stds = results_df.std()
    # statannot은 long-form이 편리하므로 변환
    df_long = results_df.copy()
    df_long = df_long.melt(var_name='method', value_name='accuracy')

    # 성능이 낮은 순으로 정렬 (예시 자료처럼 좌->우로 성능 증가하도록)
    order = list(means.sort_values().index)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='method', y='accuracy', data=df_long, order=order, capsize=.2,
                     ci='sd', palette='tab10')

    # y축 범위는 데이터에 맞게 여유를 둔다
    ymin = max(0.0, float(df_long['accuracy'].min()) - 0.02)
    ymax = min(1.01, float(df_long['accuracy'].max()) + 0.03)
    ax.set(ylim=(ymin, ymax))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    if _HAS_STATANNOT:
        # 최고 성능 모델과 나머지 모두를 비교하는 브래킷 생성
        box_pairs = [(m, best_model) for m in order if m != best_model]
        add_stat_annotation(
            ax,
            data=df_long,
            x='method',
            y='accuracy',
            order=order,
            box_pairs=box_pairs,
            test='Wilcoxon',
            text_format='star',
            comparisons_correction=None,
            loc='inside',
            verbose=0
        )
    else:
        # statannot 미설치 시, 간단한 별표 텍스트만 표기
        sorted_models = order
        for i, model in enumerate(sorted_models):
            if model != best_model and model in p_values:
                p_val = p_values[model]
                if p_val < 0.05:
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    y = means[model] + stds[model] + 0.005
                    ax.text(i, y, sig, ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel('Classification Algorithms', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

    return order, [means[m] for m in order]

def main():
    """메인 분석 함수"""
    print("=" * 60)
    print("WINE 데이터셋 교차검증 및 통계 검정 분석")
    print("=" * 60)
    
    # 1. 데이터 로드 및 전처리
    X, y = load_and_preprocess_data()
    
    # 2. 분류 알고리즘 정의
    classifiers = get_classifiers()
    
    # 3. 5-fold Cross Validation 수행
    cv_5fold_results = perform_5fold_cv(X, y, classifiers)
    results_5fold = pd.DataFrame(cv_5fold_results)
    
    # 4. Shuffle Split Cross Validation 수행
    cv_shuffle_results = perform_shuffle_split_cv(X, y, classifiers)
    results_shuffle = pd.DataFrame(cv_shuffle_results)
    
    # 5. 최고 성능 모델 찾기
    best_5fold = results_5fold.mean().idxmax()
    best_shuffle = results_shuffle.mean().idxmax()
    
    print(f"\n5-fold CV 최고 성능 모델: {best_5fold} ({results_5fold.mean()[best_5fold]:.4f})")
    print(f"Shuffle Split CV 최고 성능 모델: {best_shuffle} ({results_shuffle.mean()[best_shuffle]:.4f})")
    
    # 6. 통계 검정 수행
    p_values_5fold = perform_statistical_tests(results_5fold, best_5fold)
    p_values_shuffle = perform_statistical_tests(results_shuffle, best_shuffle)
    
    print("\n5-fold CV 통계 검정 결과 (최고 모델 vs 다른 모델들):")
    for model, p_val in p_values_5fold.items():
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{best_5fold} vs {model}: p = {p_val:.4f} {significance}")
    
    print("\nShuffle Split CV 통계 검정 결과 (최고 모델 vs 다른 모델들):")
    for model, p_val in p_values_shuffle.items():
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{best_shuffle} vs {model}: p = {p_val:.4f} {significance}")
    
    # 7. Bar plot 생성
    print("\nBar plot 생성 중...")
    
    # 5-fold CV Bar plot
    models_5fold, means_5fold = create_barplot_with_significance(
        results_5fold, p_values_5fold, best_5fold, 
        'Wine Dataset: 5-fold Cross Validation Results', 
        'wine_5fold_barplot.png'
    )
    
    # Shuffle Split CV Bar plot
    models_shuffle, means_shuffle = create_barplot_with_significance(
        results_shuffle, p_values_shuffle, best_shuffle, 
        'Wine Dataset: Shuffle Split Cross Validation Results (40 iterations)', 
        'wine_shuffle_split_barplot.png'
    )
    
    # 8. 결과 요약
    print("\n" + "=" * 60)
    print("WINE 데이터셋 교차검증 분석 결과 요약")
    print("=" * 60)
    
    print("\n1. 5-fold Cross Validation 결과:")
    print(f"   최고 성능 모델: {best_5fold} (정확도: {results_5fold.mean()[best_5fold]:.4f})")
    print("   모델별 성능 순위:")
    for i, (model, mean_acc) in enumerate(zip(models_5fold, means_5fold), 1):
        print(f"   {i}. {model}: {mean_acc:.4f}")
    
    print("\n2. Shuffle Split Cross Validation 결과 (40회):")
    print(f"   최고 성능 모델: {best_shuffle} (정확도: {results_shuffle.mean()[best_shuffle]:.4f})")
    print("   모델별 성능 순위:")
    for i, (model, mean_acc) in enumerate(zip(models_shuffle, means_shuffle), 1):
        print(f"   {i}. {model}: {mean_acc:.4f}")
    
    print("\n3. 통계적 유의성:")
    print("   - *: p < 0.05, **: p < 0.01, ***: p < 0.001")
    print("   - ns: not significant (p >= 0.05)")
    
    print("\n분석 완료! 그래프는 figures/ 폴더에 저장되었습니다.")
    
    return {
        'results_5fold': results_5fold,
        'results_shuffle': results_shuffle,
        'best_5fold': best_5fold,
        'best_shuffle': best_shuffle,
        'p_values_5fold': p_values_5fold,
        'p_values_shuffle': p_values_shuffle
    }

if __name__ == "__main__":
    results = main()
