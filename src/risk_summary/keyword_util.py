import pandas as pd
from collections import Counter
from typing import List, Tuple


EXCLUDE = ['앱-삭제', '앱-탈퇴']
KEYWORD_COL = "keywords"

# 키워드 카운팅
def keyword_count(df:pd.DataFrame) -> Counter:
    all_reviews = [k for ks in df[KEYWORD_COL] for k in ks]
    counter = Counter(all_reviews)
    
    return counter

# 키워드 비율 계산
def target_keyword_ratio(counter:Counter, target:str) -> float:
    total = sum(counter.values())
    if total == 0: # 키워드가 없다면
        return 0.0

    count = counter[target]
    return round((count / total) * 100, 2)

# 키워드 TopN
def top_n_keywords_extract(counter:Counter, n:int=3, exclude:List[str]=None):
    topn = [(k, v) for k, v in counter.most_common() if k not in exclude][:n]
    return topn

