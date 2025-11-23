#!/usr/bin/env python
"""각 모델 제거 실험 테스트"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 직접 import해서 실행
from experiments.ensemble_experiment import main

print("LogisticRegression 제거 실험 시작...")
try:
    main(case="remove_logistic", exclude_models=["LogisticRegression"])
    print("완료!")
except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()


