"""
최고 성능 분류 모델 파이프라인
EDA 결과를 기반으로 트리 기반 모델과 딥러닝 모델을 모두 구현하고 최적화
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import DataPreprocessor, create_cv_folds, load_data
from utils.evaluation import macro_f1_score, cross_validate_model, print_evaluation_report
import lightgbm as lgb
import xgboost as xgb
from models.tree_models import (
    LightGBMWrapper, XGBoostWrapper, CatBoostWrapper,
    optimize_lightgbm, optimize_xgboost, optimize_catboost
)
from models.neural_networks import (
    MLPClassifier, TabularTransformer, SensorDataset,
    optimize_mlp, optimize_transformer, train_epoch, validate
)
from models.ensemble import WeightedEnsemble, optimize_ensemble_weights
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class ModelPipeline:
    """전체 모델 파이프라인"""
    
    def __init__(self, n_trials_tree=100, n_trials_nn=100, n_trials_transformer=50, n_jobs=1):
        """
        Parameters:
        -----------
        n_trials_tree : int
            트리 모델 Optuna trial 개수
        n_trials_nn : int
            Neural Network Optuna trial 개수
        n_trials_transformer : int
            Transformer Optuna trial 개수
        n_jobs : int
            병렬 작업 개수
        """
        self.n_trials_tree = n_trials_tree
        self.n_trials_nn = n_trials_nn
        self.n_trials_transformer = n_trials_transformer
        self.n_jobs = n_jobs
        
        self.preprocessor = None
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, train_df, test_df):
        """데이터 전처리"""
        print("=" * 60)
        print("데이터 전처리 시작")
        print("=" * 60)
        
        # 전처리 파이프라인 생성 및 학습
        self.preprocessor = DataPreprocessor(
            remove_high_corr=True,
            corr_threshold=0.99,
            use_feature_selection=False
        )
        
        X_train, y_train = self.preprocessor.fit_transform(train_df)
        X_test = self.preprocessor.transform(test_df)
        
        print(f"학습 데이터 shape: {X_train.shape}")
        print(f"테스트 데이터 shape: {X_test.shape}")
        print(f"클래스 개수: {len(np.unique(y_train))}")
        
        return X_train, y_train, X_test
    
    def optimize_tree_models(self, X, y, cv_splits):
        """트리 기반 모델 최적화"""
        print("\n" + "=" * 60)
        print("트리 기반 모델 최적화 시작")
        print("=" * 60)
        
        # CV splits를 리스트로 변환 (재사용 가능하도록)
        cv_splits_list = list(cv_splits)
        
        # LightGBM 최적화
        print("\n[1/3] LightGBM 최적화...")
        try:
            lgb_params, lgb_study = optimize_lightgbm(
                X, y, iter(cv_splits_list),
                n_trials=self.n_trials_tree,
                n_jobs=self.n_jobs
            )
            self.best_params['lightgbm'] = lgb_params
            self.cv_results['lightgbm'] = lgb_study.best_value
        except Exception as e:
            print(f"LightGBM 최적화 실패: {e}")
            self.best_params['lightgbm'] = None
        
        # XGBoost 최적화
        print("\n[2/3] XGBoost 최적화...")
        try:
            xgb_params, xgb_study = optimize_xgboost(
                X, y, iter(cv_splits_list),
                n_trials=self.n_trials_tree,
                n_jobs=self.n_jobs
            )
            self.best_params['xgboost'] = xgb_params
            self.cv_results['xgboost'] = xgb_study.best_value
        except Exception as e:
            print(f"XGBoost 최적화 실패: {e}")
            self.best_params['xgboost'] = None
        
        # CatBoost 최적화
        print("\n[3/3] CatBoost 최적화...")
        try:
            cb_params, cb_study = optimize_catboost(
                X, y, iter(cv_splits_list),
                n_trials=self.n_trials_tree,
                n_jobs=self.n_jobs
            )
            self.best_params['catboost'] = cb_params
            self.cv_results['catboost'] = cb_study.best_value
        except Exception as e:
            print(f"CatBoost 최적화 실패: {e}")
            self.best_params['catboost'] = None
    
    def optimize_nn_models(self, X, y, cv_splits):
        """딥러닝 모델 최적화"""
        print("\n" + "=" * 60)
        print("딥러닝 모델 최적화 시작")
        print("=" * 60)
        
        # CV splits를 리스트로 변환 (재사용 가능하도록)
        cv_splits_list = list(cv_splits)
        
        # MLP 최적화
        print("\n[1/2] MLP 최적화...")
        try:
            mlp_params, mlp_study = optimize_mlp(
                X, y, iter(cv_splits_list),
                n_trials=self.n_trials_nn,
                device=device
            )
            self.best_params['mlp'] = mlp_params
            self.cv_results['mlp'] = mlp_study.best_value
        except Exception as e:
            print(f"MLP 최적화 실패: {e}")
            self.best_params['mlp'] = None
        
        # Transformer 최적화
        print("\n[2/2] Transformer 최적화...")
        try:
            transformer_params, transformer_study = optimize_transformer(
                X, y, iter(cv_splits_list),
                n_trials=self.n_trials_transformer,
                device=device
            )
            self.best_params['transformer'] = transformer_params
            self.cv_results['transformer'] = transformer_study.best_value
        except Exception as e:
            print(f"Transformer 최적화 실패: {e}")
            self.best_params['transformer'] = None
    
    def train_final_models(self, X, y, cv_splits):
        """최종 모델 학습 (전체 데이터 사용)"""
        print("\n" + "=" * 60)
        print("최종 모델 학습 시작")
        print("=" * 60)
        
        # 트리 모델 학습
        if self.best_params.get('lightgbm'):
            print("\n[1/5] LightGBM 최종 학습...")
            lgb_model = LightGBMWrapper(self.best_params['lightgbm'])
            train_data = lgb.Dataset(X, label=y)
            lgb_model.model = lgb.train(
                self.best_params['lightgbm'],
                train_data,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            self.models['lightgbm'] = lgb_model
        
        if self.best_params.get('xgboost'):
            print("[2/5] XGBoost 최종 학습...")
            xgb_params = self.best_params['xgboost'].copy()
            xgb_model = XGBoostWrapper(xgb_params)
            dtrain = xgb.DMatrix(X, label=y)
            xgb_model.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=1000
            )
            self.models['xgboost'] = xgb_model
        
        if self.best_params.get('catboost'):
            print("[3/5] CatBoost 최종 학습...")
            cb_params = self.best_params['catboost'].copy()
            cb_params['iterations'] = 1000
            cb_model = CatBoostWrapper(cb_params)
            cb_model.fit(X, y)
            self.models['catboost'] = cb_model
        
        # 딥러닝 모델 학습
        if self.best_params.get('mlp'):
            print("[4/5] MLP 최종 학습...")
            mlp_model = self._train_mlp_final(X, y, self.best_params['mlp'])
            self.models['mlp'] = mlp_model
        
        if self.best_params.get('transformer'):
            print("[5/5] Transformer 최종 학습...")
            transformer_model = self._train_transformer_final(X, y, self.best_params['transformer'])
            self.models['transformer'] = transformer_model
    
    def _train_mlp_final(self, X, y, params):
        """MLP 최종 학습"""
        model = MLPClassifier(
            input_dim=X.shape[1],
            num_classes=21,
            hidden_layers=params['hidden_layers'],
            hidden_units=params['hidden_units'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        dataset = SensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            train_loss, train_acc = train_epoch(
                model, dataloader, criterion, optimizer, device, scaler, device == 'cuda'
            )
            scheduler.step()
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return model
    
    def _train_transformer_final(self, X, y, params):
        """Transformer 최종 학습"""
        model = TabularTransformer(
            input_dim=X.shape[1],
            num_classes=21,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            dropout=params['dropout']
        ).to(device)
        
        dataset = SensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            train_loss, train_acc = train_epoch(
                model, dataloader, criterion, optimizer, device, scaler, device == 'cuda'
            )
            scheduler.step()
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return model
    
    def evaluate_all_models(self, X, y, cv_splits):
        """모든 모델 평가 및 최고 성능 모델 선택"""
        print("\n" + "=" * 60)
        print("모델 평가 및 비교")
        print("=" * 60)
        
        # 각 모델의 CV 성능 측정
        final_cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n[{model_name}] 교차 검증 중...")
            
            if model_name in ['mlp', 'transformer']:
                # 딥러닝 모델은 별도 평가 필요
                cv_scores = []
                for fold, (train_idx, val_idx) in enumerate(cv_splits):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # 모델 재학습
                    if model_name == 'mlp':
                        temp_model = self._train_mlp_final(X_train, y_train, self.best_params['mlp'])
                    else:
                        temp_model = self._train_transformer_final(X_train, y_train, self.best_params['transformer'])
                    
                    # 평가
                    val_dataset = SensorDataset(X_val, y_val)
                    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
                    criterion = nn.CrossEntropyLoss()
                    
                    _, f1 = validate(temp_model, val_loader, criterion, device)
                    cv_scores.append(f1)
                
                final_cv_results[model_name] = {
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores),
                    'scores': cv_scores
                }
            else:
                # 트리 모델
                results = cross_validate_model(model, X, y, cv_splits, verbose=False)
                final_cv_results[model_name] = {
                    'mean': results['mean_macro_f1'],
                    'std': results['std_macro_f1'],
                    'scores': results['scores']
                }
            
            print(f"  Macro-F1: {final_cv_results[model_name]['mean']:.6f} (+/- {final_cv_results[model_name]['std']:.6f})")
        
        # 결과 비교
        print("\n" + "-" * 60)
        print("모델 성능 비교")
        print("-" * 60)
        sorted_models = sorted(final_cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            print(f"{rank}. {model_name:15s}: {results['mean']:.6f} (+/- {results['std']:.6f})")
        
        # 최고 성능 모델 선택
        best_model_name = sorted_models[0][0]
        self.best_model = self.models[best_model_name]
        self.best_score = sorted_models[0][1]['mean']
        
        print(f"\n최고 성능 모델: {best_model_name} (Macro-F1: {self.best_score:.6f})")
        
        return final_cv_results
    
    def predict(self, X_test):
        """테스트 데이터 예측"""
        if self.best_model is None:
            raise ValueError("최고 모델이 선택되지 않았습니다. evaluate_all_models를 먼저 실행하세요.")
        
        if isinstance(self.best_model, (MLPClassifier, TabularTransformer)):
            # 딥러닝 모델
            self.best_model.eval()
            dataset = SensorDataset(X_test)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for batch_x in dataloader:
                    batch_x = batch_x.to(device)
                    outputs = self.best_model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
            
            return np.array(all_preds)
        else:
            # 트리 모델
            return self.best_model.predict(X_test)
    
    def save_models(self, save_dir='models'):
        """모델 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 전처리기 저장
        with open(f'{save_dir}/preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # 최고 모델 저장
        if isinstance(self.best_model, (MLPClassifier, TabularTransformer)):
            torch.save(self.best_model.state_dict(), f'{save_dir}/best_model.pth')
        else:
            # 트리 모델은 각 라이브러리의 저장 방법 사용
            model_name = [k for k, v in self.models.items() if v == self.best_model][0]
            if model_name == 'lightgbm':
                self.best_model.model.save_model(f'{save_dir}/best_model.txt')
            elif model_name == 'xgboost':
                self.best_model.model.save_model(f'{save_dir}/best_model.json')
            elif model_name == 'catboost':
                self.best_model.model.save_model(f'{save_dir}/best_model.cbm')
        
        # 파라미터 및 결과 저장
        results = {
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'best_model_name': [k for k, v in self.models.items() if v == self.best_model][0],
            'best_score': self.best_score
        }
        
        with open(f'{save_dir}/results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n모델 저장 완료: {save_dir}/")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("최고 성능 분류 모델 파이프라인 시작")
    print("=" * 60)
    
    # 데이터 로드
    train_df, test_df = load_data('train.csv', 'test.csv')
    
    # 파이프라인 생성
    pipeline = ModelPipeline(
        n_trials_tree=100,
        n_trials_nn=100,
        n_trials_transformer=50,
        n_jobs=1  # Optuna는 병렬 처리 시 주의 필요
    )
    
    # 데이터 전처리
    X_train, y_train, X_test = pipeline.prepare_data(train_df, test_df)
    
    # CV 스플릿 생성 (재사용을 위해 리스트로 변환)
    cv_splits = list(create_cv_folds(train_df, n_splits=5, random_state=42))
    
    # 트리 모델 최적화
    pipeline.optimize_tree_models(X_train, y_train, cv_splits)
    
    # 딥러닝 모델 최적화
    pipeline.optimize_nn_models(X_train, y_train, cv_splits)
    
    # 최종 모델 학습
    pipeline.train_final_models(X_train, y_train, cv_splits)
    
    # 모델 평가 및 선택
    final_results = pipeline.evaluate_all_models(X_train, y_train, cv_splits)
    
    # 테스트 예측
    print("\n" + "=" * 60)
    print("테스트 데이터 예측")
    print("=" * 60)
    test_predictions = pipeline.predict(X_test)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'target': test_predictions
    })
    
    os.makedirs('submissions', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f'submissions/submission_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n제출 파일 저장: {submission_path}")
    
    # 모델 저장
    pipeline.save_models('models')
    
    print("\n" + "=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()

