import ast
import sqlite3
import matplotlib as mpl
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
from collections import Counter
from typing import List, Tuple


EXCLUDE = ['앱-삭제', '앱-탈퇴']
KEYWORD_COL = "keywords"

# 한글 폰트 설정
def set_korean_font():
    mpl.rcParams["font.family"] = "NanumGothic"
    # 마이너스 기호 깨짐 방지
    mpl.rcParams["axes.unicode_minus"] = False

# db에 저장된 키워드 변환 : str->list
def parse_keywords(x):
    """
    DB에 저장된 keywords(str)를 list로 변환
    - "['a','b']" 형태 -> ['a','b']
    - "a, b" 형태 -> ['a','b']
    - None/빈값 -> []
    """
    if x is None:
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    # 리스트 문자열 형태 시도
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return [str(k).strip() for k in out if str(k).strip()]
        except Exception:
            pass

    # fallback: 콤마 분리
    return [k.strip() for k in s.split(",") if k.strip()]

# 특정 월 데이터 조회
def fetch_month_df(db_path: str, table: str, yyyymm: str) -> pd.DataFrame:
    year, month = map(int, yyyymm.split("-"))

    start_date = date(year, month, 1)
    end_date = start_date + relativedelta(months=1)

    conn = sqlite3.connect(db_path)

    if table == 'data':
        query = f"""
            SELECT *
            FROM {table}
            WHERE at >= ? AND at < ?
        """
        params = (start_date.isoformat(), end_date.isoformat())
    elif table == "summary":
        query = f"""
            SELECT *
            FROM {table}
            WHERE month = ?
        """
        params = (yyyymm,)

    df = pd.read_sql(
        query,
        conn,
        params=params
    )

    conn.close()
    return df
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
def top_n_keywords_extract(counter:Counter, n:int=3, exclude:List[str]=EXCLUDE):
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

