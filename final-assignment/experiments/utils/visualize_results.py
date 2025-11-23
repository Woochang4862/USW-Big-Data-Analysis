#!/usr/bin/env python
"""
과적합 실험 결과 시각화
- x축: 복잡도 (Simple -> Medium -> Complex -> Overfit)
- y축: Score = (F1 + AUROC) / 2
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 한글 폰트 이슈 방지

# 한글 폰트 설정 (macOS)
try:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def load_results(results_dir: Path) -> pd.DataFrame:
    """결과 CSV 파일 로드"""
    csv_path = results_dir / "overfitting_summary.csv"
    df = pd.read_csv(csv_path)
    return df


def get_complexity_order():
    """복잡도 순서 정의"""
    # Level-10부터 Level14까지 (25개 범주)
    # Decision Tree와 Neural Network는 Level1~Level5만 사용하지만, 이는 positive_levels에 이미 포함됨
    negative_levels = [f"Level{i}" for i in range(-10, 0)]
    zero_level = ["Level0"]
    positive_levels = [f"Level{i}" for i in range(1, 15)]
    return negative_levels + zero_level + positive_levels


def plot_model_complexity_trends(df: pd.DataFrame, output_dir: Path):
    """모델별 복잡도에 따른 score 변화 추이 시각화"""
    
    complexity_order = get_complexity_order()
    model_types = df["model_type"].unique()
    
    # 전체 모델을 하나의 그림에 표시
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        "LogisticRegression": "#1f77b4",
        "DecisionTree": "#ff7f0e",
        "RandomForest": "#2ca02c",
        "XGBoost": "#d62728",
        "NeuralNetwork": "#9467bd",
        "LightGBM": "#e67e22",
        "CatBoost": "#17becf",
        "Stacking": "#8B4513",
        "StackingRF": "#4682B4",
    }
    
    markers = {
        "LogisticRegression": "o",
        "DecisionTree": "s",
        "RandomForest": "^",
        "XGBoost": "D",
        "NeuralNetwork": "v",
        "LightGBM": "*",
        "CatBoost": "p",
        "Stacking": "x",
        "StackingRF": "d",
    }
    
    for model_type in model_types:
        model_df = df[df["model_type"] == model_type].copy()
        
        # 복잡도 순서에 맞게 정렬
        model_df["complexity_rank"] = model_df["config_name"].apply(
            lambda x: complexity_order.index(x) if x in complexity_order else len(complexity_order)
        )
        model_df = model_df.sort_values("complexity_rank")
        
        # 복잡도 순서에 맞는 config_name만 사용
        valid_configs = [c for c in complexity_order if c in model_df["config_name"].values]
        model_df = model_df[model_df["config_name"].isin(valid_configs)]
        
        if len(model_df) == 0:
            continue
        
        x_positions = [complexity_order.index(c) for c in model_df["config_name"]]
        
        # Train Score
        ax.plot(
            x_positions,
            model_df["train_score"],
            marker=markers[model_type],
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=colors[model_type],
            label=f"{model_type} (Train)",
            alpha=0.8
        )
        
        # Test Score
        ax.plot(
            x_positions,
            model_df["test_score"],
            marker=markers[model_type],
            linestyle="--",
            linewidth=2,
            markersize=8,
            color=colors[model_type],
            label=f"{model_type} (Test)",
            alpha=0.6
        )
    
    ax.set_xlabel("Complexity", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score = (F1 + AUROC) / 2", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance vs Complexity", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(complexity_order)))
    ax.set_xticklabels(complexity_order, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    output_path = output_dir / "complexity_trends_all.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"전체 모델 비교 그래프 저장: {output_path}")
    plt.close()
    
    # 모델별 개별 그래프 생성
    for model_type in model_types:
        model_df = df[df["model_type"] == model_type].copy()
        
        if len(model_df) == 0:
            continue
        
        # 복잡도 순서에 맞게 정렬
        model_df["complexity_rank"] = model_df["config_name"].apply(
            lambda x: complexity_order.index(x) if x in complexity_order else len(complexity_order)
        )
        model_df = model_df.sort_values("complexity_rank")
        
        valid_configs = [c for c in complexity_order if c in model_df["config_name"].values]
        model_df = model_df[model_df["config_name"].isin(valid_configs)]
        
        if len(model_df) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_positions = [complexity_order.index(c) for c in model_df["config_name"]]
        
        # Train Score
        ax.plot(
            x_positions,
            model_df["train_score"],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=10,
            color="#2c3e50",
            label="Train Score",
            alpha=0.9
        )
        
        # Test Score
        ax.plot(
            x_positions,
            model_df["test_score"],
            marker="s",
            linestyle="--",
            linewidth=2.5,
            markersize=10,
            color="#e74c3c",
            label="Test Score",
            alpha=0.9
        )
        
        # Overfitting Gap 영역 표시
        ax.fill_between(
            x_positions,
            model_df["train_score"],
            model_df["test_score"],
            alpha=0.2,
            color="#e74c3c",
            label="Overfitting Gap"
        )
        
        ax.set_xlabel("Complexity", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score = (F1 + AUROC) / 2", fontsize=12, fontweight="bold")
        ax.set_title(f"{model_type} - Performance vs Complexity", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(valid_configs)))
        ax.set_xticklabels(valid_configs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=10)
        ax.set_ylim([0.5, 1.05])
        
        plt.tight_layout()
        output_path = output_dir / f"complexity_trends_{model_type.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"{model_type} 그래프 저장: {output_path}")
        plt.close()


def plot_overfitting_gap(df: pd.DataFrame, output_dir: Path):
    """과적합 Gap 변화 추이 시각화"""
    
    complexity_order = get_complexity_order()
    model_types = df["model_type"].unique()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        "LogisticRegression": "#1f77b4",
        "DecisionTree": "#ff7f0e",
        "RandomForest": "#2ca02c",
        "XGBoost": "#d62728",
        "NeuralNetwork": "#9467bd",
        "LightGBM": "#e67e22",
        "CatBoost": "#17becf",
        "Stacking": "#8B4513",
        "StackingRF": "#4682B4"
    }
    
    for model_type in model_types:
        model_df = df[df["model_type"] == model_type].copy()
        
        # 복잡도 순서에 맞게 정렬
        model_df["complexity_rank"] = model_df["config_name"].apply(
            lambda x: complexity_order.index(x) if x in complexity_order else len(complexity_order)
        )
        model_df = model_df.sort_values("complexity_rank")
        
        valid_configs = [c for c in complexity_order if c in model_df["config_name"].values]
        model_df = model_df[model_df["config_name"].isin(valid_configs)]
        
        if len(model_df) == 0:
            continue
        
        x_positions = [complexity_order.index(c) for c in model_df["config_name"]]
        
        ax.plot(
            x_positions,
            model_df["overfitting_gap_score"],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=10,
            color=colors[model_type],
            label=model_type,
            alpha=0.8
        )
    
    ax.set_xlabel("Complexity", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overfitting Gap (Train Score - Test Score)", fontsize=12, fontweight="bold")
    ax.set_title("Overfitting Gap vs Complexity", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(complexity_order)))
    ax.set_xticklabels(complexity_order, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "overfitting_gap_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"과적합 Gap 그래프 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = base_dir / "results"
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("과적합 실험 결과 시각화")
    print("=" * 60)
    
    # 결과 로드
    print("\n결과 파일 로드 중...")
    df = load_results(results_dir)
    print(f"   총 {len(df)}개 실험 결과 로드 완료")
    
    # 시각화
    print("\n시각화 생성 중...")
    plot_model_complexity_trends(df, output_dir)
    plot_overfitting_gap(df, output_dir)
    
    print("\n" + "=" * 60)
    print("시각화 완료!")
    print(f"결과 저장 위치: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

