# 시험이 webapp/ 의 모듈(ship_core)을 찾게 합니다.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
