# 프롬프트 개선 및 성능 테스트.

import os
import json
import time
import argparse
import pandas as pd
from google import genai
from typing import Tuple
from src.dashboard.util import keyword_count, top_n_keywords_extract


# --- 1. 기타 ---

EXCEPT_KEYWORD = ['앱-삭제', '앱-탈퇴']
JSON_TEMPLATE_EX = """
{
  "situations": [],
  "evaluations": [],
  "solutions": [],
  "reason_id": []
}
""".strip()

def str_to_list(x):
    x = x.strip("[]")

    if not x:
        return []

    return [item.strip().strip('"').strip("'") for item in x.split(",") if item.strip()]


# --- 2. 프롬프트 생성 ---

def build_batch_prompt(reviews: dict, keyword) -> str:
    inputs = "\n".join([f"- id: {rid} | {content}" for rid, content in reviews.items()])
    return f"""
당신은 고객 경험(CX) 분석 전문가입니다.
제공된 리뷰 데이터는 {keyword} 키워드를 포함한 리뷰들입니다.
이 리뷰들을 분석하여 서비스 개선을 위한 인사이트를 다음 항목에 맞춰 요약해 주세요.

중요 규칙:
- 반드시 아래 [Review List]의 내용에만 근거하여 객관적으로 작성하세요.
- 출력은 반드시 **순수한 JSON 객체(Object)** 하나만 출력하세요.
- **JSON 외의 어떤 설명, 인사말, 코드블록(```)도 절대 포함하지 마세요.**
- reason_id에는 근거로 사용한 리뷰의 id를 최대 20개까지 넣으세요(Review List의 id 그대로).

작성 가이드라인:
- 각 배열 항목은 불렛이 아니라, JSON 문자열 리스트로 작성하세요.
- 가치 판단/비난 없이 사용자 의견을 객관적으로 정리하세요.

### 1. 'situations' 
- [{keyword}] 관련 주요 이탈 원인
- 사용자가 해당 문제를 경험할 때 주로 발생하는 구체적인 상황들을 나열하세요.
- 단순한 현상 외에, 그 문제가 발생함으로써 유발되는 2차적인 불편함이나 감정적 불쾌감을 포함하세요.

### 2. 'evaluations'
- 문제 발생 시 기존 대응에 대한 평가
- 현재 시스템이나 고객센터, 가게 측의 대응(보상, 안내 방식, 소통 창구 등)에 대해 사용자들이 느끼는 솔직한 피드백을 요약하세요.
- 사용자가 '부족하다'고 느끼거나 '오히려 기분이 나빴다'고 언급한 지점이 어디인지 명확히 짚어주세요.

### 3. 'solutions'
- 사용자들이 원하는 근본적인 해결 방안
- 리뷰어들이 직접 제안하거나, 불만 내용에서 유추할 수 있는 실질적인 개선책을 정리하세요.
- (예: 시스템 정책 변경, 패널티 강화, 보상 체계 현실화, 실시간 소통 강화 등)

### 4. 'reason_id'
- 위 분석의 근거가 된 리뷰 ID 리스트를 반환하세요. (최대 20개)

[Review List]
{inputs}

[출력 예시]
{JSON_TEMPLATE_EX}

""".strip()


# --- 3. 리뷰 요약 함수 ---

def llm_summary_reviews(reviews: dict, keyword:str, model:str="gemini-2.0-flash") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GEMINI_API_KEY를 설정해 주세요. ($env:GEMINI_API_KEY=...)")

    client = genai.Client(api_key=api_key)

    # 리뷰 전체를 한 번에 입력
    prompt = build_batch_prompt(reviews, keyword)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    return json.loads(resp.text)

# --- 4. 요약 파이프라인 ---

# case 분류
# 1. "불만","확정" 클래스 모두 없는 경우 => '없음'만 리뷰 요약 => 어떤 점이 강점인지(아직 미구현)
# 2. "불만" 클래스가 없는 경우 => '확정'만 리뷰 요약
# 3. "확정" 클래스가 없는 경우 => '불만'만 리뷰 요약
# 4. 타겟 키워드가 "불만" 클래스에 없는 경우 => 각 클래스별 top 키워드로 요약
# 5. "불만","확정" 클래스 모두 존재. 타겟 키워드가 "불만" 클래스에 존재

# 우선 5번 케이스만 구현하기
def summary_pipeline(df:pd.DataFrame, model:str="gemini-2.0-flash") -> Tuple[dict, dict]:
    # 클래스 분리
    df_positive = df[df["churn_intent_label"] == 0].copy()
    df_complaint = df[df["churn_intent_label"] == 1].copy()
    df_confirmed = df[df["churn_intent_label"] == 2].copy()

    # api 호출 시 입력으로 들어갈 상위 리뷰 500개 추출
    df_complaint = df_complaint.sort_values(by=['thumbsUpCount', 'at'], ascending=[False, False]).head(500)
    df_confirmed = df_confirmed.sort_values(by=['thumbsUpCount', 'at'], ascending=[False, False]).head(500)

    # '확정' top 키워드 추출
    counter = keyword_count(df_confirmed)
    topn = top_n_keywords_extract(counter)
    target = topn[0][0]

    # 여기서 케이스 분기점 만들어야 함. (추후 예정)
    # 타겟 키워드 포함하는 리뷰 추출
    review_complaint = {rid: content for rid, content, ks in zip(df_complaint['reviewId'], df_complaint["content"], df_complaint["keywords"]) if target in ks}
    review_confirmed = {rid: content for rid, content, ks in zip(df_confirmed['reviewId'], df_confirmed["content"], df_confirmed["keywords"]) if target in ks}

    # 리뷰 요약
    summary_complaint = llm_summary_reviews(review_complaint, target, model)
    summary_confirmed = llm_summary_reviews(review_confirmed, target, model)

    return summary_complaint, summary_confirmed


# --- 5. main ---

def main():
    p = argparse.ArgumentParser(description="동기식 LLM 리뷰요약")
    p.add_argument("--csv", required=True)
    p.add_argument("--model", default="gemini-2.0-flash") # gemini-2.5-flash, gemini-3-flash-preview

    args = p.parse_args()
    
    # 데이터 로드
    df = pd.read_csv(args.csv)
    df['keywords'] = df['keywords'].map(str_to_list)
    print(df['churn_intent'].value_counts())
    print("모델:", args.model)
    
    # 리뷰 요약
    start_time = time.time()
    summary_complaint, summary_confirmed = summary_pipeline(df, args.model)
    end_time = time.time()
    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print("['불만' 리뷰 요약]")
    print(summary_complaint)
    print("\n['확정' 리뷰 요약]")
    print(summary_confirmed)

if __name__ == "__main__":
    main()
