#!/usr/bin/env python3
"""SAP 모델 접근 권한 확인 스크립트"""

from huggingface_hub import HfApi

api = HfApi()

try:
    model_info = api.model_info('SAP/sap-rpt-1-oss')
    print("✓ 모델 접근 권한이 있습니다!")
    print(f"모델: SAP/sap-rpt-1-oss")
    print(f"Gated: {getattr(model_info, 'gated', 'Unknown')}")
except Exception as e:
    print(f"✗ 모델 접근 권한이 없습니다: {e}")
    print("\n" + "="*60)
    print("⚠️  모델 접근 권한을 요청해야 합니다!")
    print("="*60)
    print("\n다음 단계를 따라주세요:")
    print("1. 브라우저에서 다음 URL 방문:")
    print("   https://huggingface.co/SAP/sap-rpt-1-oss")
    print("\n2. 'Request access' 또는 'Request to access' 버튼 클릭")
    print("\n3. 접근 권한이 승인될 때까지 대기 (보통 몇 분~몇 시간 소요)")
    print("\n4. 권한이 승인된 후 다시 실행")
    print("="*60)

