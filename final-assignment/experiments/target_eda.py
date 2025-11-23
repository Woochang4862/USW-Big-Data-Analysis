"""
타깃 변수 HE_D3_label 중심 EDA
비타민D 충분(1) vs 결핍(0) 예측 문제 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
data_dir = Path('data/processed')
df = pd.read_csv(data_dir / 'combined_dataset.csv')

print("=" * 80)
print("타깃 변수 HE_D3_label 중심 EDA")
print("=" * 80)

# 1. 타깃 변수 기본 통계
print("\n[1] 타깃 변수 분포")
print("-" * 80)
target_dist = df['HE_D3_label'].value_counts().sort_index()
target_pct = df['HE_D3_label'].value_counts(normalize=True).sort_index() * 100
print(f"클래스 0 (비타민D 결핍): {target_dist[0]:,}명 ({target_pct[0]:.2f}%)")
print(f"클래스 1 (비타민D 충분): {target_dist[1]:,}명 ({target_pct[1]:.2f}%)")
print(f"총 샘플 수: {len(df):,}명")
print(f"클래스 불균형 비율: {target_dist[1]/target_dist[0]:.2f}:1")

# Train/Test 분할 확인
print("\n[2] Train/Test 분할별 타깃 분포")
print("-" * 80)
split_target = pd.crosstab(df['split'], df['HE_D3_label'], margins=True)
print(split_target)
split_target_pct = pd.crosstab(df['split'], df['HE_D3_label'], normalize='index') * 100
print("\n비율 (%)")
print(split_target_pct)

# 3. 성별/연령대별 타깃 분포
print("\n[3] 성별/연령대별 타깃 분포")
print("-" * 80)
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100], labels=['19-30', '31-50', '51-65', '65+'])
sex_age_cross = pd.crosstab([df['sex'], df['age_group']], df['HE_D3_label'], margins=True)
print(sex_age_cross)
sex_age_pct = pd.crosstab([df['sex'], df['age_group']], df['HE_D3_label'], normalize='index') * 100
print("\n비율 (%)")
print(sex_age_pct)

# 4. 주요 임상 지표와 타깃 변수 관계
print("\n[4] 주요 임상 지표별 타깃 변수 분포")
print("-" * 80)
clinical_vars = {
    'BMI': 'HE_BMI',
    '공복혈당': 'HE_glu',
    'HbA1c': 'HE_HbA1c',
    '총콜레스테롤': 'HE_chol',
    'HDL': 'HE_HDL_st2',
    '중성지방': 'HE_TG',
    'LDL': 'HE_LDL_drct',
    '헤모글로빈': 'HE_HB',
    '크레아티닌': 'HE_crea'
}

for name, col in clinical_vars.items():
    if col in df.columns:
        stats = df.groupby('HE_D3_label')[col].agg(['mean', 'std', 'median'])
        print(f"\n{name} ({col}):")
        print(stats)

# 5. 만성질환과 타깃 변수 관계
print("\n[5] 만성질환 유병 여부와 타깃 변수 관계")
print("-" * 80)
disease_vars = {
    '고혈압': 'DI1_pr',
    '당뇨병': 'DE1_pr',
    '이상지질혈증': 'DI2_pr',
    '관절염': 'DM1_pr',
    '우울증': 'DF2_pr'
}

for name, col in disease_vars.items():
    if col in df.columns:
        # -1은 비해당이므로 제외하고 분석
        disease_cross = pd.crosstab(
            df[df[col].isin([0, 1])][col],
            df[df[col].isin([0, 1])]['HE_D3_label'],
            margins=True
        )
        print(f"\n{name} ({col}):")
        print(disease_cross)
        if len(disease_cross) > 2:
            disease_pct = pd.crosstab(
                df[df[col].isin([0, 1])][col],
                df[df[col].isin([0, 1])]['HE_D3_label'],
                normalize='index'
            ) * 100
            print("비율 (%)")
            print(disease_pct)

# 6. 건강행태와 타깃 변수 관계
print("\n[6] 건강행태와 타깃 변수 관계")
print("-" * 80)
behavior_vars = {
    '현재흡연': 'BS3_1',
    '월간음주': 'dr_month',
    '유산소활동': 'pa_aerobic',
    '스트레스인지': 'BP1'
}

for name, col in behavior_vars.items():
    if col in df.columns:
        # 결측 제외
        valid_data = df[df[col].notna() & df[col] != -1]
        if len(valid_data) > 0:
            behavior_cross = pd.crosstab(valid_data[col], valid_data['HE_D3_label'], margins=True)
            print(f"\n{name} ({col}):")
            print(behavior_cross.head(10))  # 상위 10개만 출력

# 7. 영양 섭취와 타깃 변수 관계
print("\n[7] 영양 섭취량과 타깃 변수 관계")
print("-" * 80)
nutrition_vars = {
    '1일 에너지': 'N_EN',
    '단백질': 'N_PROT',
    '지방': 'N_FAT',
    '탄수화물': 'N_CHO',
    '식이섬유': 'N_TDF',
    '나트륨': 'N_NA',
    '칼슘': 'N_CA',
    '비타민C': 'N_VITC',
    '비타민D': 'N_VITD'
}

for name, col in nutrition_vars.items():
    if col in df.columns:
        stats = df.groupby('HE_D3_label')[col].agg(['mean', 'std', 'median'])
        print(f"\n{name} ({col}):")
        print(stats)

# 8. 결측 패턴 분석
print("\n[8] 타깃 변수별 주요 변수 결측률")
print("-" * 80)
key_vars = ['HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_TG', 'N_EN', 'N_PROT', 'N_VITD']
missing_by_target = {}
for var in key_vars:
    if var in df.columns:
        missing_by_target[var] = df.groupby('HE_D3_label')[var].apply(lambda x: x.isna().mean() * 100)
        
missing_df = pd.DataFrame(missing_by_target).T
missing_df.columns = ['결핍(0)', '충분(1)']
print(missing_df)

# 9. 시각화 저장
print("\n[9] 시각화 생성 중...")
print("-" * 80)

fig_dir = Path('results/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

# 9-1. 타깃 변수 분포
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('타깃 변수 HE_D3_label 분석', fontsize=16, fontweight='bold')

# 분포 막대그래프
ax1 = axes[0, 0]
target_dist.plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4'])
ax1.set_title('타깃 변수 분포', fontsize=12, fontweight='bold')
ax1.set_xlabel('HE_D3_label (0:결핍, 1:충분)')
ax1.set_ylabel('빈도')
ax1.set_xticklabels(['결핍 (0)', '충분 (1)'], rotation=0)
for i, v in enumerate(target_dist):
    ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Train/Test 분할별 분포
ax2 = axes[0, 1]
split_target_pct.plot(kind='bar', ax=ax2, color=['#ff6b6b', '#4ecdc4'])
ax2.set_title('Train/Test 분할별 타깃 분포', fontsize=12, fontweight='bold')
ax2.set_xlabel('데이터 분할')
ax2.set_ylabel('비율 (%)')
ax2.legend(['결핍 (0)', '충분 (1)'])
ax2.set_xticklabels(split_target_pct.index, rotation=0)

# 성별/연령대별 분포
ax3 = axes[1, 0]
sex_age_pct_plot = sex_age_pct.reset_index()
sex_age_pct_plot['group'] = sex_age_pct_plot['sex'].astype(str) + '_' + sex_age_pct_plot['age_group'].astype(str)
x_pos = np.arange(len(sex_age_pct_plot))
width = 0.35
ax3.bar(x_pos - width/2, sex_age_pct_plot[0], width, label='결핍 (0)', color='#ff6b6b')
ax3.bar(x_pos + width/2, sex_age_pct_plot[1], width, label='충분 (1)', color='#4ecdc4')
ax3.set_title('성별/연령대별 타깃 분포', fontsize=12, fontweight='bold')
ax3.set_xlabel('성별_연령대')
ax3.set_ylabel('비율 (%)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(sex_age_pct_plot['group'], rotation=45, ha='right')
ax3.legend()

# 주요 임상 지표 박스플롯
ax4 = axes[1, 1]
plot_vars = ['HE_BMI', 'HE_glu', 'HE_HbA1c']
plot_data = []
for var in plot_vars:
    if var in df.columns:
        for label in [0, 1]:
            values = df[(df['HE_D3_label'] == label) & (df[var].notna())][var].values
            plot_data.extend([{'변수': var, '타깃': label, '값': v} for v in values[:500]])  # 샘플링
            
plot_df = pd.DataFrame(plot_data)
if len(plot_df) > 0:
    sns.boxplot(data=plot_df, x='변수', y='값', hue='타깃', ax=ax4)
    ax4.set_title('주요 임상 지표별 타깃 분포', fontsize=12, fontweight='bold')
    ax4.set_ylabel('값')
    ax4.legend(title='타깃', labels=['결핍 (0)', '충분 (1)'])

plt.tight_layout()
plt.savefig(fig_dir / 'target_analysis_overview.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'target_analysis_overview.png'}")

# 9-2. 상관관계 히트맵
fig, ax = plt.subplots(figsize=(12, 10))
corr_vars = ['HE_D3_label', 'age', 'sex', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 
             'HE_chol', 'HE_TG', 'HE_HDL_st2', 'N_EN', 'N_VITD', 'N_CA']
corr_data = df[corr_vars].select_dtypes(include=[np.number])
corr_matrix = corr_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('타깃 변수와 주요 변수 간 상관관계', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(fig_dir / 'target_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'target_correlation_heatmap.png'}")

# 9-3. 연령대별 타깃 분포
fig, ax = plt.subplots(figsize=(10, 6))
age_target = pd.crosstab(df['age_group'], df['HE_D3_label'], normalize='index') * 100
age_target.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'], width=0.8)
ax.set_title('연령대별 비타민D 상태 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('연령대')
ax.set_ylabel('비율 (%)')
ax.set_xticklabels(age_target.index, rotation=0)
ax.legend(['결핍 (0)', '충분 (1)'], title='타깃')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'target_by_age_group.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'target_by_age_group.png'}")

# 9-4. 영양 섭취와 타깃 변수
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('영양 섭취량과 타깃 변수 관계', fontsize=16, fontweight='bold')

nut_vars_plot = [('N_EN', '1일 에너지 (kcal)'), ('N_VITD', '비타민D (μg)'), 
                 ('N_CA', '칼슘 (mg)'), ('N_VITC', '비타민C (mg)')]

for idx, (var, title) in enumerate(nut_vars_plot):
    ax = axes[idx // 2, idx % 2]
    if var in df.columns:
        data_0 = df[(df['HE_D3_label'] == 0) & (df[var].notna())][var]
        data_1 = df[(df['HE_D3_label'] == 1) & (df[var].notna())][var]
        
        ax.hist(data_0, bins=30, alpha=0.6, label='결핍 (0)', color='#ff6b6b', density=True)
        ax.hist(data_1, bins=30, alpha=0.6, label='충분 (1)', color='#4ecdc4', density=True)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('섭취량')
        ax.set_ylabel('밀도')
        ax.legend()
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'target_nutrition_distribution.png', dpi=300, bbox_inches='tight')
print(f"저장: {fig_dir / 'target_nutrition_distribution.png'}")

print("\n" + "=" * 80)
print("EDA 완료!")
print("=" * 80)

