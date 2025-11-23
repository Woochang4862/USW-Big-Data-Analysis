#!/usr/bin/env python3
"""SAP 모델 다운로드 테스트"""

import os
from huggingface_hub import hf_hub_download, login, whoami

print("="*60)
print("SAP 모델 다운로드 테스트")
print("="*60)

# 현재 로그인 상태 확인
try:
    user_info = whoami()
    print(f"✓ 현재 로그인된 사용자: {user_info.get('name', 'Unknown')}")
except Exception as e:
    print(f"✗ 로그인 확인 실패: {e}")
    print("\n토큰을 확인합니다...")
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print("✓ 환경 변수에서 토큰 발견")
        try:
            login(token=token, add_to_git_credential=False)
            print("✓ 토큰으로 로그인 성공")
        except Exception as login_err:
            print(f"✗ 로그인 실패: {login_err}")
    else:
        print("✗ 환경 변수에 토큰이 없습니다")
        print("\n해결 방법:")
        print("1. huggingface-cli login 실행")
        print("2. 또는 export HF_TOKEN='your_token' 설정")

print("\n모델 파일 다운로드 시도...")
try:
    # 토큰 가져오기
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    
    path = hf_hub_download(
        repo_id='SAP/sap-rpt-1-oss',
        filename='2025-11-04_sap-rpt-one-oss.pt',
        token=token
    )
    print(f"✓ 다운로드 성공!")
    print(f"  경로: {path}")
except Exception as e:
    print(f"✗ 다운로드 실패: {e}")
    print("\n" + "="*60)
    print("문제 해결 방법:")
    print("="*60)
    print("1. Hugging Face에 로그인 확인:")
    print("   huggingface-cli whoami")
    print("\n2. 토큰을 환경 변수에 설정:")
    print("   export HF_TOKEN='your_token_here'")
    print("\n3. 모델 접근 권한 확인:")
    print("   https://huggingface.co/SAP/sap-rpt-1-oss")
    print("   'Request access' 버튼이 있으면 클릭")
    print("="*60)

