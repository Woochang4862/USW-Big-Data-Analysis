#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교차검증 결과 시각화 및 통계 검정
Wilcoxon 검정을 통한 모델 간 유의성 분석
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac의 경우
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 디렉토리 설정
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# JSON 파일 로드
with open(os.path.join(RESULTS_DIR, 'cv_results.json'), 'r') as f:
    cv_results = json.load(f)

# 데이터 변환 - wine_plot_statannot.py 스타일
# 각 모델의 test_scores를 dict로 변환
test_results = {}
train_results = {}

for model_name, results in cv_results['cv_results'].items():
    test_results[model_name] = results['test_scores']
    train_results[model_name] = results['train_scores']

# DataFrame 생성
df_test = pd.DataFrame(test_results)
df_train = pd.DataFrame(train_results)

# 통계 출력
print("="*60)
print("모델별 정확도 통계")
print("="*60)
print("\n[Test Accuracy]")
print(df_test.mean().sort_values(ascending=False))
print("\n[Test Accuracy - Std]")
print(df_test.std().sort_values())
print("\n[Train Accuracy]")
print(df_train.mean().sort_values(ascending=False))
print("="*60)

# 1. Test Accuracy 플롯 (Wilcoxon 검정 포함) - wine_plot_statannot.py 스타일
def plot_with_statannot(df: pd.DataFrame, title: str, filename: str, ylabel: str = 'Accuracy'):
    """wine_plot_statannot.py 스타일의 플롯 생성"""
    means = df.mean()
    order = list(means.sort_values().index)
    long = df.melt(var_name='method', value_name='accuracy')
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        x='method', 
        y='accuracy', 
        data=long, 
        capsize=.2, 
        order=order, 
        ci='sd', 
        palette='tab10'
    )
    
    # 참고 그림 스케일에 맞춰 상단 여백 확보
    min_val = long['accuracy'].min()
    max_val = long['accuracy'].max()
    ax.set(ylim=(min_val - 0.02, min(1.01, max_val + 0.12)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    
    # 최고 성능 모델과 나머지 전부 비교
    if len(order) >= 2:
        box_pairs = [(m, order[-1]) for m in order if m != order[-1]]
        try:
            test_results = add_stat_annotation(
                ax, 
                data=long, 
                x='method', 
                y='accuracy', 
                order=order,
                box_pairs=box_pairs, 
                test='Wilcoxon', 
                text_format='star',
                comparisons_correction=None, 
                loc='inside', 
                verbose=2
            )
            print(f"\n[{title} - Wilcoxon 검정 결과]")
            print(test_results)
        except Exception as e:
            print(f"\nWilcoxon 검정 수행 중 오류 발생: {e}")
    
    plt.xlabel('Classification Algorithms', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'\n그림 저장 완료: {out}')
    plt.show()

# Test Accuracy 플롯 (Wilcoxon 검정 포함)
plot_with_statannot(
    df_test, 
    '교차검증 Test Accuracy 비교 (Wilcoxon 검정)', 
    'test_accuracy_comparison_wilcoxon.png',
    'Test Accuracy'
)

# Train Accuracy 플롯
plot_with_statannot(
    df_train, 
    '교차검증 Train Accuracy 비교', 
    'train_accuracy_comparison.png',
    'Train Accuracy'
)

# 2. Train vs Test 비교 플롯
print("\n" + "="*60)
print("Train vs Test 비교 플롯 생성 중...")
print("="*60)

# 데이터 변환
data_list = []
for model_name in df_test.columns:
    for i in range(len(df_test)):
        data_list.append({
            'method': model_name,
            'fold': i + 1,
            'accuracy_train': df_train.loc[i, model_name],
            'accuracy_test': df_test.loc[i, model_name]
        })

df_results = pd.DataFrame(data_list)

# 모델별 평균 정확도 계산 (정렬용)
model_mean_test = df_results.groupby('method')['accuracy_test'].mean().sort_values()
sorted_models = model_mean_test.index.tolist()

fig, ax = plt.subplots(figsize=(14, 8))

df_melted = df_results.melt(
    id_vars=['method', 'fold'],
    value_vars=['accuracy_train', 'accuracy_test'],
    var_name='dataset',
    value_name='accuracy'
)
df_melted['dataset'] = df_melted['dataset'].map({
    'accuracy_train': 'Train',
    'accuracy_test': 'Test'
})

sns.barplot(
    data=df_melted,
    x='method',
    y='accuracy',
    hue='dataset',
    order=sorted_models,
    capsize=.2,
    ci='sd',
    palette={'Train': '#3498db', 'Test': '#e74c3c'}
)

min_acc = df_melted['accuracy'].min()
max_acc = df_melted['accuracy'].max()
y_range = max_acc - min_acc
ax.set(ylim=(max(0, min_acc - y_range * 0.1), min(1, max_acc + y_range * 0.2)))

ax.set_xlabel('Classification Algorithms', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('교차검증 Train vs Test Accuracy 비교', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.legend(title='Dataset', fontsize=11, title_fontsize=12)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'train_vs_test_accuracy_comparison.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'그림 저장 완료: {out_path}')
plt.show()

print("\n" + "="*60)
print("모든 분석 완료!")
print("="*60)
