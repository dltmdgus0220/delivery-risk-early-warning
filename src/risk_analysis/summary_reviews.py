import pandas as pd
from google import genai
from typing import Tuple, List, Optional
import argparse
import time
import re
from collections import Counter
import os


EXCEPT_KEYWORD = ['앱-삭제', '앱-탈퇴']

# --- 1. 기타 함수 ---

def str_to_list_keyword(keywords: str) -> List[str]:
    keywords = keywords.strip('[]')
    if len(keywords) == 0:
        return []
    else:
        return [x.strip() for x in keywords.replace("'", "").split(",")]

def clean_one_line(s:str) -> str:
    s = (s or "").strip() # 양쪽 공백 제거
    s = s.strip().strip('"').strip("'").strip() # 양쪽 따옴표 제거
    s = s.replace("\n", " ").replace("\r", " ") # 줄바꿈->공백 변환. 한줄로 결과 바꾸기
    s = re.sub(r"\s+", " ", s).strip() # 연속되는 여러 개의 공백 변환
    s = re.sub(r"^[\-\*\d\.\)\s]+", "", s).strip() # 불릿/번호 형태 제거
    return s


# --- 2. 프롬프트 생성 ---

def build_batch_prompt(texts: List[str], keyword) -> str:
    text_inputs = "\n".join([
        f"ID_{i+1}: {t}" for i, t in enumerate(texts)
    ]) 
    return f"""
너는 고객 리뷰 데이터를 분석하는 데이터 분석가다.
주어진 리뷰 목록을 바탕으로 공통적으로 나타나는 불만 포인트를 "명사구"로 추출하라.
리뷰에 없는 내용을 추측하거나 과장하지 마라.

[제한 사항]
- 반드시 "{keyword}" 키워드가 포함된 리뷰들의 맥락에 맞춰 요약하라.
- 출력은 정확히 2~3개 항목만 포함하라.
- 각 항목은 2~10자 내의 짧은 명사구(예: "불친절한 고객센터")로 작성하라.
- 항목은 쉼표(,)로만 구분하고 따옴표/불릿/번호 없이 출력하라.
    
[Review List]
{text_inputs}

[입력 예시]
"ID_1: 고객센터 교육 다시 해라"
"ID_2: 배달이 잘못 와서 전화했는데 고작 3천원 쿠폰주고 끝이다"
"ID_3: 상담원이 너무 불친절하고 문제를 해결해줄 의지가 없음."

[출력 예시]
불친절한 고객센터, 불만족스러운 보상

""".strip()


# --- 3. 리뷰 요약 함수 ---

def llm_summary_reviews(keyword:str, lst: List[str], model:str="gemini-2.0-flash") -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GEMINI_API_KEY를 설정해 주세요. ($env:GEMINI_API_KEY=...)")

    client = genai.Client(api_key=api_key)

    # 리뷰 전체를 한 번에 입력
    prompt = build_batch_prompt(lst, keyword)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    resp = clean_one_line(getattr(resp, "text", ""))
    return resp


# --- 4. 리뷰 전처리 ---

def select_target_keyword_and_reviews(df:pd.DataFrame, target:str | None=None) -> Tuple[str, List[str]]:
    # top 키워드 도출
    def _top_keyword(d: pd.DataFrame, exclude: List[str]) -> Optional[str]:
        kws = [k for ks in d["keywords"] for k in ks if k not in exclude]
        if not kws:
            return None
        return Counter(kws).most_common(1)[0][0]

    # 타겟 키워드 포함한 리뷰 도출
    def _collect_reviews(d: pd.DataFrame, kw: str) -> List[str]:
        if kw is None or d.empty:
            return []
        mask = d["keywords"].apply(lambda ks: kw in ks)
        return d.loc[mask, "content"].dropna().astype(str).tolist()
    
    if target:
        reviews = _collect_reviews(df, target)
    else:
        target = _top_keyword(df, EXCEPT_KEYWORD)
        reviews = _collect_reviews(df, target)

    return target, reviews

