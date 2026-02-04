import pandas as pd
from typing import List
from collections import Counter
from src.risk_detection.risk_score_calc import monthly_risk_calc
from src.risk_detection.summary_reviews import summary_pipeline, str_to_list_keyword, EXCEPT_KEYWORD


# --- 1. 기타 함수 ---

# 키워드 비율 계산
def keyword_ratio(df:pd.DataFrame, target:str) -> float:
    all_keywords = [k for ks in df["keywords"] for k in ks]
    total = len(all_keywords)
    if total == 0:
        return 0.0

    count = sum(1 for k in all_keywords if k == target)
    return round((count / total) * 100, 2)

