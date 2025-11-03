#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, 'results_5fold.json'), 'r') as f:
    res5 = json.load(f)
with open(os.path.join(RESULTS_DIR, 'results_shuffle.json'), 'r') as f:
    resh = json.load(f)

def plot_with_statannot(results: dict, title: str, filename: str):
    df = pd.DataFrame(results)
    means = df.mean()
    order = list(means.sort_values().index)
    long = df.melt(var_name='method', value_name='accuracy_test')

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='method', y='accuracy_test', data=long, capsize=.2, order=order, ci='sd', palette='tab10')
    # 참고 그림 스케일에 맞춰 상단 여백 확보
    ax.set(ylim=(long['accuracy_test'].min()-0.02, min(1.01, long['accuracy_test'].max()+0.12)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    # 최고 성능 모델과 나머지 전부 비교
    box_pairs = [(m, order[-1]) for m in order if m != order[-1]]
    add_stat_annotation(
        ax, data=long, x='method', y='accuracy_test', order=order,
        box_pairs=box_pairs, test='Wilcoxon', text_format='star',
        comparisons_correction=None, loc='inside', verbose=2
    )

    plt.xlabel('Classification Algorithms', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print('Saved:', out)

plot_with_statannot(res5, 'Wine Dataset: 5-fold Cross Validation Results (statannot)', 'wine_5fold_barplot_statannot.png')
plot_with_statannot(resh, 'Wine Dataset: Shuffle Split Cross Validation Results (40 iterations, statannot)', 'wine_shuffle_split_barplot_statannot.png')
