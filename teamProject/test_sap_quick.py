#!/usr/bin/env python3
"""SAP 모델 테스트 스크립트 - 빠른 검증용"""

import sys
import os

print("="*60)
print("SAP 모델 테스트 시작")
print("="*60)

# 1. 모듈 import 테스트
print("\n[1/5] 모듈 import 테스트...")
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from sap_rpt_oss import SAP_RPT_OSS_Classifier
    print("✓ 모든 모듈 import 성공")
except Exception as e:
    print(f"✗ 모듈 import 실패: {e}")
    sys.exit(1)

# 2. 데이터 로드 테스트
print("\n[2/5] 데이터 로드 테스트...")
try:
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_full = df_train.drop(['ID', 'target'], axis=1)
    y = df_train['target'].astype(int)
    test_ids = df_test['ID']
    X_test_full = df_test.drop(['ID'], axis=1)
    print(f"✓ 데이터 로드 성공: Train={X_full.shape[0]}, Test={X_test_full.shape[0]}, Features={X_full.shape[1]}, Classes={len(np.unique(y))}")
except Exception as e:
    print(f"✗ 데이터 로드 실패: {e}")
    sys.exit(1)

# 3. 포트 충돌 방지 테스트
print("\n[3/5] 포트 충돌 방지 테스트...")
try:
    import socket
    import subprocess
    
    def check_port_in_use(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    ZMQ_PORT = 5655
    port_status = check_port_in_use(ZMQ_PORT)
    print(f"✓ 포트 {ZMQ_PORT} 상태 확인: {'사용 중' if port_status else '비어있음'}")
except Exception as e:
    print(f"⚠️  포트 확인 중 오류: {e}")

# 4. Hugging Face 인증 테스트
print("\n[4/5] Hugging Face 인증 테스트...")
try:
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"✓ Hugging Face 로그인 확인: {user_info.get('name', 'Unknown')}")
except Exception as e:
    print(f"⚠️  Hugging Face 인증 확인 실패: {e}")

# 5. SAP 모델 초기화 테스트 (실제 학습은 하지 않음)
print("\n[5/5] SAP 모델 초기화 테스트...")
try:
    # 작은 샘플로 테스트
    X_sample = X_full.head(100)
    y_sample = y.head(100)
    
    SAP_PARAMS = {
        'max_context_size': 512,  # 테스트용으로 작게 설정
        'bagging': 2  # 테스트용으로 작게 설정
    }
    
    print(f"  파라미터: {SAP_PARAMS}")
    print("  모델 초기화 중... (시간이 걸릴 수 있습니다)")
    
    # 모델 초기화만 테스트 (fit은 하지 않음)
    clf = SAP_RPT_OSS_Classifier(**SAP_PARAMS)
    print("✓ SAP 모델 초기화 성공")
    print("  (실제 학습은 시간이 오래 걸리므로 여기서 중단)")
    
except Exception as e:
    print(f"✗ SAP 모델 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ 모든 테스트 통과!")
print("="*60)
print("\n다음 단계:")
print("1. 전체 데이터로 학습하려면: python sap.py")
print("2. 학습 시간이 오래 걸릴 수 있습니다 (수십 분~수시간)")
print("3. 메모리 부족 시 max_context_size와 bagging 값을 줄이세요")

