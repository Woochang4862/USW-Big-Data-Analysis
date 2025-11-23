"""
데이터 전처리 및 Feature Engineering 모듈
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, remove_high_corr=True, corr_threshold=0.99, use_feature_selection=False, top_n_features=30):
        """
        Parameters:
        -----------
        remove_high_corr : bool
            높은 상관관계를 가진 피처 제거 여부
        corr_threshold : float
            상관관계 임계값 (이 값보다 높으면 제거)
        use_feature_selection : bool
            Feature Importance 기반 피처 선택 사용 여부
        top_n_features : int
            선택할 피처 개수
        """
        self.remove_high_corr = remove_high_corr
        self.corr_threshold = corr_threshold
        self.use_feature_selection = use_feature_selection
        self.top_n_features = top_n_features
        
        self.scaler = None
        self.feature_cols = None
        self.selected_features = None
        self.high_corr_features_to_remove = None
        
    def fit(self, train_df):
        """
        전처리 파이프라인 학습
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            학습 데이터
        """
        # Feature 컬럼 추출
        self.feature_cols = [col for col in train_df.columns if col.startswith('X_')]
        
        # 높은 상관관계 피처 제거
        if self.remove_high_corr:
            self._remove_high_correlation_features(train_df[self.feature_cols])
            print(f"높은 상관관계 피처 제거: {len(self.high_corr_features_to_remove)}개")
        
        # 사용할 피처 목록 결정
        if self.high_corr_features_to_remove:
            self.selected_features = [f for f in self.feature_cols 
                                   if f not in self.high_corr_features_to_remove]
        else:
            self.selected_features = self.feature_cols.copy()
        
        # Scaler 학습
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[self.selected_features])
        
        print(f"최종 사용 피처 개수: {len(self.selected_features)}")
        
    def transform(self, df):
        """
        데이터 변환
        
        Parameters:
        -----------
        df : pd.DataFrame
            변환할 데이터
            
        Returns:
        --------
        X : np.ndarray
            변환된 피처 배열
        """
        X = self.scaler.transform(df[self.selected_features])
        return X
    
    def fit_transform(self, train_df):
        """
        학습 및 변환
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            학습 데이터
            
        Returns:
        --------
        X : np.ndarray
            변환된 피처 배열
        y : np.ndarray (선택적)
            타겟 배열
        """
        self.fit(train_df)
        X = self.transform(train_df)
        
        if 'target' in train_df.columns:
            y = train_df['target'].values
            return X, y
        return X
    
    def _remove_high_correlation_features(self, df):
        """
        높은 상관관계를 가진 피처 제거
        
        Parameters:
        -----------
        df : pd.DataFrame
            피처만 포함된 데이터프레임
        """
        corr_matrix = df.corr().abs()
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if upper_triangle[i, j] and corr_matrix.iloc[i, j] >= self.corr_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # 제거할 피처 선택 (각 쌍에서 나중에 나타나는 피처 제거)
        self.high_corr_features_to_remove = []
        for feat1, feat2, corr_val in high_corr_pairs:
            # 두 피처 중 하나만 제거 (보통 두 번째 피처)
            if feat2 not in self.high_corr_features_to_remove:
                self.high_corr_features_to_remove.append(feat2)


def create_cv_folds(train_df, n_splits=5, random_state=42):
    """
    Stratified K-Fold 교차 검증 스플릿 생성
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        학습 데이터
    n_splits : int
        폴드 개수
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    skf : StratifiedKFold
        교차 검증 스플릿터
    """
    feature_cols = [col for col in train_df.columns if col.startswith('X_')]
    X = train_df[feature_cols].values
    y = train_df['target'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y)


def load_data(train_path='train.csv', test_path='test.csv'):
    """
    데이터 로드
    
    Parameters:
    -----------
    train_path : str
        학습 데이터 경로
    test_path : str
        테스트 데이터 경로
        
    Returns:
    --------
    train_df : pd.DataFrame
        학습 데이터
    test_df : pd.DataFrame
        테스트 데이터
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"학습 데이터 shape: {train_df.shape}")
    print(f"테스트 데이터 shape: {test_df.shape}")
    
    return train_df, test_df

