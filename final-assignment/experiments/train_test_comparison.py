"""
Train/Test 데이터셋 분포 비교 분석
두 데이터셋이 비슷한 특징을 띄는지 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
data_dir = Path('data/processed')
df = pd.read_csv(data_dir / 'combined_dataset.csv')

print("=" * 80)
print("Train/Test 데이터셋 분포 비교 분석")
print("=" * 80)

# Train/Test 분할
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

print(f"\n데이터 크기:")
print(f"Train: {len(train_df):,}명")
print(f"Test: {len(test_df):,}명")
print(f"비율: {len(train_df)/len(df)*100:.1f}% / {len(test_df)/len(df)*100:.1f}%")

# 1. 타깃 변수 분포 비교
print("\n" + "=" * 80)
print("[1] 타깃 변수 분포 비교")
print("=" * 80)

target_train = train_df['HE_D3_label'].value_counts(normalize=True).sort_index() * 100
target_test = test_df['HE_D3_label'].value_counts(normalize=True).sort_index() * 100

print("\n타깃 변수 분포 (비율 %):")
comparison_df = pd.DataFrame({
    'Train': target_train,
    'Test': target_test,
    '차이': target_test - target_train
})
print(comparison_df)

# 카이제곱 검정
chi2, p_value = stats.chi2_contingency(
    pd.crosstab(df['split'], df['HE_D3_label'])
)[:2]
print(f"\n카이제곱 검정: χ² = {chi2:.4f}, p-value = {p_value:.4f}")
if p_value > 0.05:
    print("→ Train/Test 간 타깃 분포에 유의한 차이 없음 (p > 0.05)")
else:
    print("→ Train/Test 간 타깃 분포에 유의한 차이 있음 (p ≤ 0.05)")

# 2. 인구통계 변수 비교
print("\n" + "=" * 80)
print("[2] 인구통계 변수 비교")
print("=" * 80)

demographic_vars = {
    '연령': 'age',
    '성별': 'sex',
    '소득4분위': 'incm',
    '교육수준': 'edu',
    '지역': 'region'
}

for name, col in demographic_vars.items():
    if col not in df.columns:
        continue
    
    print(f"\n{name} ({col}):")
    
    if df[col].dtype in ['int64', 'float64']:
        # 연속형 변수
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()
        
        stats_df = pd.DataFrame({
            'Train': [train_vals.mean(), train_vals.std(), train_vals.median()],
            'Test': [test_vals.mean(), test_vals.std(), test_vals.median()]
        }, index=['평균', '표준편차', '중앙값'])
        print(stats_df)
        
        # t-test
        if len(train_vals) > 0 and len(test_vals) > 0:
            t_stat, p_val = stats.ttest_ind(train_vals, test_vals)
            print(f"t-test: t = {t_stat:.4f}, p-value = {p_val:.4f}")
            if p_val > 0.05:
                print("→ 유의한 차이 없음")
            else:
                print("→ 유의한 차이 있음")
    else:
        # 범주형 변수
        train_dist = train_df[col].value_counts(normalize=True).sort_index() * 100
        test_dist = test_df[col].value_counts(normalize=True).sort_index() * 100
        
        comp_df = pd.DataFrame({
            'Train (%)': train_dist,
            'Test (%)': test_dist,
            '차이': test_dist - train_dist
        })
        print(comp_df)
        
        # 카이제곱 검정
        crosstab = pd.crosstab(df['split'], df[col])
        if crosstab.shape[0] == 2 and crosstab.shape[1] > 1:
            chi2, p_val = stats.chi2_contingency(crosstab)[:2]
            print(f"카이제곱 검정: χ² = {chi2:.4f}, p-value = {p_val:.4f}")
            if p_val > 0.05:
                print("→ 유의한 차이 없음")
            else:
                print("→ 유의한 차이 있음")

# 3. 주요 임상 지표 비교
print("\n" + "=" * 80)
print("[3] 주요 임상 지표 비교")
print("=" * 80)

clinical_vars = {
    'BMI': 'HE_BMI',
    '공복혈당': 'HE_glu',
    'HbA1c': 'HE_HbA1c',
    '총콜레스테롤': 'HE_chol',
    'HDL': 'HE_HDL_st2',
    '중성지방': 'HE_TG',
    'LDL': 'HE_LDL_drct'
}

clinical_comparison = []
for name, col in clinical_vars.items():
    if col not in df.columns:
        continue
    
    train_vals = train_df[col].dropna()
    test_vals = test_df[col].dropna()
    
    if len(train_vals) > 0 and len(test_vals) > 0:
        train_mean = train_vals.mean()
        test_mean = test_vals.mean()
        diff = test_mean - train_mean
        diff_pct = (diff / train_mean * 100) if train_mean != 0 else 0
        
        t_stat, p_val = stats.ttest_ind(train_vals, test_vals)
        
        clinical_comparison.append({
            '변수': name,
            'Train_평균': train_mean,
            'Test_평균': test_mean,
            '차이': diff,
            '차이(%)': diff_pct,
            'p-value': p_val,
            '유의차': '없음' if p_val > 0.05 else '있음'
        })

clinical_df = pd.DataFrame(clinical_comparison)
print(clinical_df.to_string(index=False))

# 4. 영양 섭취량 비교
print("\n" + "=" * 80)
print("[4] 영양 섭취량 비교")
print("=" * 80)

nutrition_vars = {
    '1일 에너지': 'N_EN',
    '단백질': 'N_PROT',
    '지방': 'N_FAT',
    '탄수화물': 'N_CHO',
    '칼슘': 'N_CA',
    '비타민D': 'N_VITD',
    '비타민C': 'N_VITC'
}

nutrition_comparison = []
for name, col in nutrition_vars.items():
    if col not in df.columns:
        continue
    
    train_vals = train_df[col].dropna()
    test_vals = test_df[col].dropna()
    
    if len(train_vals) > 0 and len(test_vals) > 0:
        train_mean = train_vals.mean()
        test_mean = test_vals.mean()
        diff = test_mean - train_mean
        diff_pct = (diff / train_mean * 100) if train_mean != 0 else 0
        
        t_stat, p_val = stats.ttest_ind(train_vals, test_vals)
        
        nutrition_comparison.append({
            '변수': name,
            'Train_평균': train_mean,
            'Test_평균': test_mean,
            '차이': diff,
            '차이(%)': diff_pct,
            'p-value': p_val,
            '유의차': '없음' if p_val > 0.05 else '있음'
        })

nutrition_df = pd.DataFrame(nutrition_comparison)
print(nutrition_df.to_string(index=False))

# 5. 결측률 비교
print("\n" + "=" * 80)
print("[5] 결측률 비교")
print("=" * 80)

key_vars = ['HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_TG', 
            'N_EN', 'N_PROT', 'N_VITD', 'N_CA']

missing_comparison = []
for var in key_vars:
    if var in df.columns:
        train_missing = train_df[var].isna().mean() * 100
        test_missing = test_df[var].isna().mean() * 100
        diff = test_missing - train_missing
        
        missing_comparison.append({
            '변수': var,
            'Train_결측률(%)': train_missing,
            'Test_결측률(%)': test_missing,
            '차이': diff
        })

missing_df = pd.DataFrame(missing_comparison)
print(missing_df.to_string(index=False))

# 6. 시각화
print("\n" + "=" * 80)
print("[6] 시각화 생성 중...")
print("=" * 80)

fig_dir = Path('results/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

# 6-1. 타깃 변수 분포 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Train/Test 타깃 변수 분포 비교', fontsize=16, fontweight='bold')

ax1 = axes[0]
target_train_counts = train_df['HE_D3_label'].value_counts().sort_index()
target_test_counts = test_df['HE_D3_label'].value_counts().sort_index()

x = np.arange(2)
width = 0.35
ax1.bar(x - width/2, target_train_counts, width, label='Train', color='#3498db', alpha=0.8)
ax1.bar(x + width/2, target_test_counts, width, label='Test', color='#e74c3c', alpha=0.8)
ax1.set_xlabel('HE_D3_label (0:결핍, 1:충분)')
ax1.set_ylabel('빈도')
ax1.set_xticks(x)
ax1.set_xticklabels(['결핍 (0)', '충분 (1)'])
ax1.legend()
ax1.set_title('빈도 비교', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
target_train_pct = train_df['HE_D3_label'].value_counts(normalize=True).sort_index() * 100
target_test_pct = test_df['HE_D3_label'].value_counts(normalize=True).sort_index() * 100

x = np.arange(2)
ax2.bar(x - width/2, target_train_pct, width, label='Train', color='#3498db', alpha=0.8)
ax2.bar(x + width/2, target_test_pct, width, label='Test', color='#e74c3c', alpha=0.8)
ax2.set_xlabel('HE_D3_label (0:결핍, 1:충분)')
ax2.set_ylabel('비율 (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(['결핍 (0)', '충분 (1)'])
ax2.legend()
ax2.set_title('비율 비교', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 값 표시
for i, (train_val, test_val) in enumerate(zip(target_train_pct, target_test_pct)):
    ax2.text(i - width/2, train_val, f'{train_val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, test_val, f'{test_val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / 'train_test_target_comparison.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'train_test_target_comparison.png'}")

# 6-2. 주요 변수 분포 비교 (히스토그램)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Train/Test 주요 변수 분포 비교', fontsize=16, fontweight='bold')

plot_vars = [
    ('age', '연령'),
    ('HE_BMI', 'BMI'),
    ('HE_glu', '공복혈당'),
    ('HE_chol', '총콜레스테롤'),
    ('HE_HDL_st2', 'HDL'),
    ('N_EN', '1일 에너지')
]

for idx, (var, title) in enumerate(plot_vars):
    ax = axes[idx // 3, idx % 3]
    
    if var in df.columns:
        train_vals = train_df[var].dropna()
        test_vals = test_df[var].dropna()
        
        if len(train_vals) > 0 and len(test_vals) > 0:
            ax.hist(train_vals, bins=30, alpha=0.6, label='Train', color='#3498db', density=True)
            ax.hist(test_vals, bins=30, alpha=0.6, label='Test', color='#e74c3c', density=True)
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            ax.set_xlabel('값')
            ax.set_ylabel('밀도')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # 통계 정보 표시
            train_mean = train_vals.mean()
            test_mean = test_vals.mean()
            t_stat, p_val = stats.ttest_ind(train_vals, test_vals)
            ax.text(0.05, 0.95, f'Train: {train_mean:.2f}\nTest: {test_mean:.2f}\np={p_val:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / 'train_test_distribution_comparison.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'train_test_distribution_comparison.png'}")

# 6-3. 통계적 유의성 요약
fig, ax = plt.subplots(figsize=(12, 8))

# p-value 비교
all_comparisons = []
all_comparisons.extend(clinical_comparison)
all_comparisons.extend(nutrition_comparison)

if len(all_comparisons) > 0:
    comp_df = pd.DataFrame(all_comparisons)
    comp_df = comp_df.sort_values('p-value')
    
    y_pos = np.arange(len(comp_df))
    colors = ['green' if p > 0.05 else 'red' for p in comp_df['p-value']]
    
    ax.barh(y_pos, comp_df['p-value'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_df['변수'])
    ax.set_xlabel('p-value')
    ax.set_title('Train/Test 통계적 유의성 검정 결과', fontsize=14, fontweight='bold')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # p-value 표시
    for i, (idx, row) in enumerate(comp_df.iterrows()):
        ax.text(row['p-value'], i, f"{row['p-value']:.4f}", 
               va='center', ha='left' if row['p-value'] < 0.5 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / 'train_test_statistical_significance.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'train_test_statistical_significance.png'}")

# 7. 종합 요약
print("\n" + "=" * 80)
print("[7] 종합 요약")
print("=" * 80)

# 유의한 차이가 있는 변수 개수
significant_vars = []
if len(clinical_comparison) > 0:
    sig_clinical = [c for c in clinical_comparison if c['p-value'] <= 0.05]
    significant_vars.extend([c['변수'] for c in sig_clinical])

if len(nutrition_comparison) > 0:
    sig_nutrition = [n for n in nutrition_comparison if n['p-value'] <= 0.05]
    significant_vars.extend([n['변수'] for n in sig_nutrition])

print(f"\n유의한 차이가 있는 변수 (p ≤ 0.05): {len(significant_vars)}개")
if len(significant_vars) > 0:
    print("변수 목록:")
    for var in significant_vars:
        print(f"  - {var}")
else:
    print("→ 모든 주요 변수에서 Train/Test 간 유의한 차이 없음")

print(f"\n유의한 차이가 없는 변수 (p > 0.05): {len(all_comparisons) - len(significant_vars)}개")

# 결론
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

if len(significant_vars) == 0:
    print("✓ Train과 Test 데이터셋이 매우 유사한 특징을 띄고 있습니다.")
    print("✓ 데이터 분할이 잘 이루어졌으며, 모델 학습에 적합합니다.")
else:
    print(f"⚠ Train과 Test 데이터셋 간 {len(significant_vars)}개 변수에서 유의한 차이가 발견되었습니다.")
    print("⚠ 모델 학습 시 이 차이를 고려해야 할 수 있습니다.")
    print("⚠ 하지만 대부분의 변수는 유사한 분포를 보이므로, 일반적인 모델링이 가능합니다.")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)


