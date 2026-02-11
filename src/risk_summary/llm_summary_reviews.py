import os
import json
import time
import argparse
import pandas as pd
from google import genai
from typing import Tuple, List
from collections import Counter

KEYWORD_COL = "keywords"
EXCLUDE = ["앱-삭제", "앱-탈퇴"]

JSON_TEMPLATE_EX = """
{
  "situations": [
    {"text": "", "importance": 0},
    {"text": "", "importance": 0}
  ],
  "evaluations": [
    {"text": "", "importance": 0},
    {"text": "", "importance": 0}
  ],
  "solutions": [
    {"text": "", "importance": 0},
    {"text": "", "importance": 0}
  ],
  "reason_id": []
}
""".strip()


def str_to_list(x):
    if pd.isna(x):
        return []
    x = str(x).strip().strip("[]")
    if not x:
        return []
    return [item.strip().strip('"').strip("'") for item in x.split(",") if item.strip()]


def keyword_count(df: pd.DataFrame) -> Counter:
    all_reviews = [k for ks in df[KEYWORD_COL] for k in ks]
    return Counter(all_reviews)


def top_n_keywords_extract(counter: Counter, n: int = 3, exclude: List[str] = EXCLUDE):
    return [(k, v) for k, v in counter.most_common() if k not in exclude][:n]


def safe_json_loads(text: str) -> dict:
    t = (text or "").strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(t[start:end + 1])


def enforce_top2(out: dict) -> dict:
    # situations / evaluations / solutions: importance 내림차순 2개 (없으면 앞 2개)
    for k in ["situations", "evaluations", "solutions"]:
        v = out.get(k, [])
        if not isinstance(v, list):
            out[k] = []
            continue

        # dict 형태면 importance 정렬
        if v and isinstance(v[0], dict):
            cleaned = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                try:
                    imp = int(item.get("importance", 0))
                except Exception:
                    imp = 0
                if text:
                    imp = max(1, min(5, imp))
                    cleaned.append({"text": text, "importance": imp})
            cleaned.sort(key=lambda x: x["importance"], reverse=True)
            out[k] = cleaned[:2]
        else:
            # 문자열 리스트면 앞 2개만 dict로 변환
            s2 = [str(x).strip() for x in v if str(x).strip()][:2]
            out[k] = [{"text": s, "importance": 3} for s in s2]

    rid = out.get("reason_id", [])
    if isinstance(rid, list):
        out["reason_id"] = [str(x) for x in rid][:20]
    else:
        out["reason_id"] = []

    return out


def to_korean_view(summary: dict) -> dict:
    """
    대시보드/콘솔에서 바로 쓰기 좋은 한글 키로 변환
    - 문제상황 / 기존 대응 / 소비자 원하는 대응 / reason_id
    """
    return {
        "문제상황": summary.get("situations", []),
        "기존 대응": summary.get("evaluations", []),
        "소비자 원하는 대응": summary.get("solutions", []),
        "reason_id": summary.get("reason_id", []),
    }


def build_batch_prompt(reviews: dict, keyword: str) -> str:
    lines = []
    for rid, content in reviews.items():
        safe_content = str(content).replace("\n", " ").strip()
        lines.append(f"id={rid} :: {safe_content}")
    inputs = "\n".join(lines)

    return f"""
당신은 고객 경험(CX) 분석 전문가입니다.
아래 리뷰들은 모두 '{keyword}' 키워드를 포함한 실제 사용자 리뷰입니다.
반드시 리뷰 내용에만 근거하여 JSON으로 요약하세요.

중요 규칙(반드시 준수):
- 반드시 아래 [Review List] 내용에만 근거하세요. (추측/일반론/외부지식 금지)
- 출력은 반드시 **순수한 JSON 객체(Object)** 1개만 출력하세요.
- JSON 외의 어떤 설명, 인사말, 코드블록(```)도 절대 포함하지 마세요.
- situations / evaluations / solutions 는 각각 **정확히 2개 항목만** 출력하세요. (덜/더 출력 금지)
- 각 항목은 반드시 {{"text":"...", "importance": n}} 형태로 작성하세요.
- importance는 1~5 정수이며, 5가 가장 중요합니다.
- 각 섹션에서 importance가 높은 순으로 2개만 선택하세요.
- reason_id에는 근거 리뷰 id를 최대 20개까지 넣으세요. (Review List의 id 그대로)

중요도(importance) 판단 기준 (반드시 준수):
- 5점(매우 높음): 1시간 이상 지연 또는 주문 취소/미도착, 명시적 이탈 표현(삭제/탈퇴/다신 안 씀),
  강한 분노/기만/배신감 표현, 동일 문제가 여러 리뷰에서 반복적으로 등장
- 4점(높음): 30~60분 지연, 음식 식음/품질 저하, 고객센터 연결 불가, 강한 불만
- 3점(보통): 반복되는 불편, 정책/보상 불만(누적 시 위험)
- 2점(낮음): 일회성 불편, 개인적 불만(감정 강도 낮음)
- 1점(매우 낮음): 정보성 의견 또는 참고 수준

항목 정의:
1) situations (문제상황) - 중요도 높은 2개
2) evaluations (기존 대응에 대한 평가) - 중요도 높은 2개
3) solutions (소비자 원하는 대응) - 중요도 높은 2개
4) reason_id - 최대 20개

[Review List]
{inputs}

[출력 예시]
{JSON_TEMPLATE_EX}
""".strip()


def llm_summary_reviews(reviews: dict, keyword: str, model: str = "gemini-2.0-flash") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GEMINI_API_KEY를 설정해 주세요. (PowerShell: $env:GEMINI_API_KEY='...')")

    client = genai.Client(api_key=api_key)
    prompt = build_batch_prompt(reviews, keyword)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    out = safe_json_loads(resp.text)
    out = enforce_top2(out)

    # ✅ 추가 안전장치: reason_id는 실제로 입력된 리뷰 id 안에서만 허용
    valid_ids = set(reviews.keys())
    out["reason_id"] = [rid for rid in out.get("reason_id", []) if rid in valid_ids][:20]

    return out


def summary_pipeline(df: pd.DataFrame, model: str = "gemini-2.0-flash") -> Tuple[dict, dict, str]:
    df_complaint = df[df["churn_intent_label"] == 1].copy()
    df_confirmed = df[df["churn_intent_label"] == 2].copy()

    df_complaint = df_complaint.sort_values(by=["thumbsUpCount", "at"], ascending=[False, False]).head(500)
    df_confirmed = df_confirmed.sort_values(by=["thumbsUpCount", "at"], ascending=[False, False]).head(500)

    counter = keyword_count(df_confirmed)
    topn = top_n_keywords_extract(counter, n=3)
    if not topn:
        raise RuntimeError("확정 리뷰에서 키워드를 추출하지 못했습니다. keywords 컬럼/전처리를 확인해 주세요.")
    target = topn[0][0]

    review_complaint = {
        rid: content
        for rid, content, ks in zip(df_complaint["reviewId"], df_complaint["content"], df_complaint["keywords"])
        if target in ks
    }
    review_confirmed = {
        rid: content
        for rid, content, ks in zip(df_confirmed["reviewId"], df_confirmed["content"], df_confirmed["keywords"])
        if target in ks
    }

    if not review_confirmed:
        raise RuntimeError(f"확정 리뷰에서 target='{target}' 포함 리뷰가 0건입니다.")

    summary_complaint = llm_summary_reviews(review_complaint, target, model)
    summary_confirmed = llm_summary_reviews(review_confirmed, target, model)

    return summary_complaint, summary_confirmed, target


def main():
    p = argparse.ArgumentParser(description="동기식 LLM 리뷰요약")
    p.add_argument("--csv", required=True)
    p.add_argument("--model", default="gemini-2.0-flash")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["keywords"] = df["keywords"].map(str_to_list)

    if "churn_intent" in df.columns:
        print(df["churn_intent"].value_counts())
    else:
        print(df["churn_intent_label"].value_counts())

    print("모델:", args.model)

    start_time = time.time()
    summary_complaint, summary_confirmed, keyword = summary_pipeline(df, args.model)
    end_time = time.time()

    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

    # ✅ 출력
    print("\n['불만' 리뷰 요약]")
    print(to_korean_view(summary_complaint))

    print("\n['확정' 리뷰 요약]")
    print(to_korean_view(summary_confirmed))


if __name__ == "__main__":
    main()