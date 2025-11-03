"""
PyTorch 기반 Neural Network 및 Transformer 모델 구현 및 Optuna 최적화
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import macro_f1_score
import warnings
warnings.filterwarnings('ignore')


class SensorDataset(Dataset):
    """센서 데이터셋"""
    
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron 분류기"""
    
    def __init__(self, input_dim, num_classes=21, hidden_layers=3, hidden_units=256, 
                 dropout_rate=0.3, use_batch_norm=True):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_units
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TabularTransformer(nn.Module):
    """Tabular Transformer 모델"""
    
    def __init__(self, input_dim, num_classes=21, d_model=128, n_heads=8, 
                 n_layers=3, dropout=0.1, max_seq_len=52):
        super(TabularTransformer, self).__init__()
        
        # Feature를 sequence로 변환
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.input_dim = input_dim
        self.d_model = d_model
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # 각 feature를 sequence로 변환
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        x = self.input_projection(x)  # (batch_size, input_dim, d_model)
        
        # Position encoding 추가
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, input_dim, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = macro_f1_score(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, macro_f1


def optimize_mlp(X, y, cv_splits, n_trials=100, timeout=None, device='cuda'):
    """
    MLP 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿
    n_trials : int
        Optuna trial 개수
    timeout : float, optional
        최적화 시간 제한 (초)
    device : str
        디바이스 ('cuda' or 'cpu')
        
    Returns:
    --------
    best_params : dict
        최적 하이퍼파라미터
    study : optuna.Study
        Optuna study 객체
    """
    def objective(trial):
        # 하이퍼파라미터 제안
        hidden_layers = trial.suggest_int('hidden_layers', 2, 5)
        hidden_units = trial.suggest_int('hidden_units', 64, 512)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 데이터 로더 생성
            train_dataset = SensorDataset(X_train, y_train)
            val_dataset = SensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # 모델 생성
            model = MLPClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_layers=hidden_layers,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate
            ).to(device)
            
            # Loss, Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            
            # Mixed precision training
            scaler = GradScaler()
            use_amp = device == 'cuda'
            
            # Early stopping
            best_f1 = 0
            patience = 10
            patience_counter = 0
            max_epochs = 100
            
            for epoch in range(max_epochs):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, scaler, use_amp
                )
                val_loss, val_f1 = validate(model, val_loader, criterion, device)
                
                scheduler.step()
                
                # Early stopping
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                
                # 중간 보고 (Pruner를 위해)
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            cv_scores.append(best_f1)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10),
        study_name='mlp_optimization'
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    print(f"\nMLP 최적화 완료!")
    print(f"Best Macro-F1: {study.best_value:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, study


def optimize_transformer(X, y, cv_splits, n_trials=50, timeout=None, device='cuda'):
    """
    Transformer 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X : np.ndarray
        피처 배열
    y : np.ndarray
        타겟 배열
    cv_splits : iterator
        교차 검증 스플릿
    n_trials : int
        Optuna trial 개수
    timeout : float, optional
        최적화 시간 제한 (초)
    device : str
        디바이스 ('cuda' or 'cpu')
        
    Returns:
    --------
    best_params : dict
        최적 하이퍼파라미터
    study : optuna.Study
        Optuna study 객체
    """
    def objective(trial):
        # 하이퍼파라미터 제안
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        n_heads = trial.suggest_int('n_heads', 4, 16)
        n_layers = trial.suggest_int('n_layers', 2, 6)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 데이터 로더 생성
            train_dataset = SensorDataset(X_train, y_train)
            val_dataset = SensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # 모델 생성
            model = TabularTransformer(
                input_dim=input_dim,
                num_classes=num_classes,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout
            ).to(device)
            
            # Loss, Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            
            # Mixed precision training
            scaler = GradScaler()
            use_amp = device == 'cuda'
            
            # Early stopping
            best_f1 = 0
            patience = 10
            patience_counter = 0
            max_epochs = 100
            
            for epoch in range(max_epochs):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, scaler, use_amp
                )
                val_loss, val_f1 = validate(model, val_loader, criterion, device)
                
                scheduler.step()
                
                # Early stopping
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                
                # 중간 보고
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            cv_scores.append(best_f1)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10),
        study_name='transformer_optimization'
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    print(f"\nTransformer 최적화 완료!")
    print(f"Best Macro-F1: {study.best_value:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, study

