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

# 새로 등장한 키워드 / 급증한 키워드
def detect_keyword_changes(counter_prev:Counter, counter_cur:Counter, threshold:float=0.1, min_cur_count:int=5) -> Tuple[List[dict], List[dict]]:
    prev_keyword = set(counter_prev.keys())
    cur_keyword = set(counter_cur.keys())

    prev_total = sum(counter_prev.values())
    cur_total = sum(counter_cur.values())

    if prev_total == 0 or cur_total == 0:
        return [], []

    all_keyword = (prev_keyword | cur_keyword)
    new = []
    surged = []

    for k in all_keyword:
        prev_cnt = counter_prev.get(k, 0)
        cur_cnt = counter_cur.get(k, 0)
        
        prev_ratio = prev_cnt / prev_total
        cur_ratio = cur_cnt / cur_total
        diff = cur_ratio - prev_ratio

        # 신규 키워드
        if prev_cnt == 0:
            new.append({
                "keyword":k,
                "cur_ratio": cur_ratio,
                "cur_count": cur_cnt
            })
            continue
        
        # 너무 적은 키워드는 노이즈 취급
        if cur_cnt < min_cur_count:
            continue
        # 급증 키워드
        if diff >= threshold:
            surged.append({
                "keyword": k,
                "prev_ratio": prev_ratio,
                "cur_ratio": cur_ratio,
                "diff_pp": diff,
                "prev_count": prev_cnt,
                "cur_count": cur_cnt
            })

    # 비중 기준 정렬
    new.sort(key=lambda x: x["ratio"], reverse=True)
    # 증가폭 기준 정렬
    surged.sort(key=lambda x: x["diff_pp"], reverse=True)
    return new, surged

