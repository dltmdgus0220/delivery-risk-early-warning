import pandas as pd
import argparse
from sklearn.metrics import classification_report


# --- 0. 설정 ---
p = argparse.ArgumentParser(description="LLM 프롬프트 평가")
p.add_argument("--dir", required=True, help="폴더 경로")

args = p.parse_args()
    
