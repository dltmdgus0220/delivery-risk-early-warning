import re
import argparse
import json
import os
import time
from typing import Any, List

import pandas as pd
from google import genai  # pip install -U google-genai


STATUS = ["지연", "느림", "빠름", "최악", "불만", "만족", "나쁨", "좋음", "많음", "적음", "없음", "부족", "미흡",
          "친절", "불친절", "태도불량", "품질불량", "작동불량", "편함", "불편", "불가", "제한", "오류", "취소",
          "비쌈", "저렴", "해지", "삭제", "탈퇴", "건의", "개선요청"]

# 배민 고유 서비스
SERVICE_KEYWORDS = ["한집배달", "가게배달", "알뜰배달", "배민1", "배민스토어", "B마트", "배민클럽", "배민패스"]

def build_batch_prompt(texts: List[str], ratings: List[int]) -> str:
    # LLM이 실수없이 잘 이해할 수 있도록 "ID_1: (별점 5점) 맛있어요" 형식으로 묶음.
    combined_inputs = "\n".join([
        f"ID_{i+1}: (별점 {r}점) {t}" for i, (r, t) in enumerate(zip(ratings, texts))
    ])
    
    return f"""
너는 숙련된 고객 경험 분석가이자 데이터 라벨러다. 아래 가이드라인을 바탕으로 배달 앱 리뷰 {len(texts)}개를 분석하라.

### [분류 가이드라인]
1. 0 (없음): 긍정적 경험. 불만이 있더라도 재이용 의사가 높거나 단순 건의인 경우.
2. 1 (약함): 서비스에 실망했으나 "다음에는 잘해달라"는 개선 요구가 있거나, 단순 불평에 그친 경우.
3. 2 (강함): 재이용 거부 의사 표명("다시는 안 함", "삭제", "돈 아깝다") 또는 위생/태도 등 치명적 결함. 반복된 불만 표출 포함.

### [분류 및 키워드 규칙]
- **비꼬기 주의**: 별점은 낮으나 내용은 긍정적(예: "진짜 빨리 오네요ㅋㅋ")인 경우 문맥상 비꼬는 것(강함)으로 판단하고, 키워드는 실제 의미(예: "배달-지연")로 추출하라.
- **과거 경험 언급**: "저번에도 이러더니 이번에도 그러네요"처럼 반복된 불만이 보이면 즉시 '강함'으로 분류하라.
- **고유 서비스명 보존**: 리뷰 내용 중 **{SERVICE_KEYWORDS}**와 관련된 언급이 있다면, 키워드 추출 시 일반적인 '배달' 대신 해당 고유 명칭을 반드시 사용하라.
- **키워드 형식**: 반드시 '[대상]-[상태]' 형태의 짧은 단어로 추출하라. 
- **상태값 제한**: 하이픈(-) 뒤의 상태값은 반드시 아래 리스트에 정의된 단어만 사용하라.
[STATUS 리스트]: {STATUS}
(예: "배달-지연", "위생-품질불량", "업데이트-불편", "배달비-비쌈", "한집배달-불만")

### [복합 감정 처리 가이드라인]
리뷰에 긍정과 부정이 섞여 있을 경우 아래의 '부정 우선순위'에 따라 라벨링하라.

1. **결정적 부정 (2: 강함)**: 
   - 서비스나 품질에 대한 칭찬이 있더라도 재이용 거부 의사, 치명적 결함, 반복 불만 중 하나라도 포함되는 경우. (예: "다시는 안 먹음", "돈 아깝다", "위생 불량", "태도 불량")
2. **부분적 부정 (1: 약함)**: 
   - 전반적으로 만족하나 특정 부분(배달, 요청사항 미이행, 가격 등)에 명확한 실망을 표현한 경우.
3. **단순 아쉬움 (0: 없음)**: 
   - 만족도가 높으며, 서비스 이탈로 이어질 가능성이 없는 가벼운 피드백.

- **키워드 반영**: 복합 감정인 경우, 만족한 부분과 불만족한 부분을 모두 키워드로 추출하라. 
  (예: "맛은 있는데 너무 늦어요" -> keywords: ["음식-좋음", "배달-지연"])

### [제약 사항]
- 반드시 제공된 ID_1부터 ID_{len(texts)}까지 순서대로 누락 없이 라벨링하라.
- 결과는 반드시 마크다운 없이 JSON 리스트 형식으로만 출력하라.
- 입력 개수와 출력 결과의 개수가 반드시 일치해야 한다.

### [입력 데이터]
{combined_inputs}

### [출력 형식 및 예시]
[
  {{
    "id": 1,
    "churn_intent": "강함",
    "churn_intent_label": 2,
    "churn_intent_confidence": 0.9,
    "reason": "재이용 의사 없음을 명확히 밝히고 배달지연에 대한 문제를 반복적으로 경험하고 있음",
    "keywords": ["배달-지연", "대응-미흡", "보상-없음"]
  }}
]
""".strip()


def extract_json(s: str) -> Any:
    s = (s or "").strip()
    # print(f"DEBUG: {s}")
    
    # 1) 마크다운 코드 블록 제거 (```json 또는 ``` 제거)
    s = re.sub(r"```json\s*|```", "", s).strip()
    
    # 2) 대괄호 [ ] 또는 중괄호 { } 사이의 내용만 추출
    # 배치 처리 시에는 리스트([])로 올 확률이 높으므로 둘 다 대응
    start_idx = s.find("[")
    end_idx = s.rfind("]")
    
    if start_idx == -1 or end_idx == -1:
        # 리스트가 아니라 단일 객체일 경우 대비
        start_idx = s.find("{")
        end_idx = s.rfind("}")

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(f"유효한 JSON 형식을 찾을 수 없습니다. 응답 내용: {s[:100]}...")
        
    json_str = s[start_idx : end_idx + 1]
    return json.loads(json_str)


def main():
    p = argparse.ArgumentParser(description="LLM(Gemini)로 CSV 일부 자동 라벨링")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--score-col", default="score", help="별점 컬럼명")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--n", type=int, default=200, help="라벨링할 샘플 수")
    p.add_argument("--batch", type=int, default=50, help="한 번에 처리할 샘플 수")
    p.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini 모델명") # gemini-1.5-flash


    args = p.parse_args()
    
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.score_col]).copy()
    df_sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    client = genai.Client() # $env:GEMINI_API_KEY='AIzaSy어쩌구'

    out_churn_intent, out_churn_intent_label, out_churn_intent_confidence, out_reason, out_keywords = [], [], [], [], []
    batch_size = args.batch # 200까지는 안정적
    
    start = time.time()
    for i in range(0, len(df_sample), batch_size):
        batch_slice = df_sample.iloc[i : i + batch_size]
        batch_texts = batch_slice[args.text_col].astype(str).tolist()
        batch_ratings = batch_slice[args.score_col].astype(int).tolist()

        prompt = build_batch_prompt(batch_texts, batch_ratings)
        print(f"[{i+1}/{len(df_sample)}] 배치 준비")
        
        success = False
        try:
            # retry 로직 포함
            success = False
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=args.model, contents=prompt, config={"temperature": 0.1})
                    data = extract_json(resp.text)
                    
                    if isinstance(data, list) and len(data) == len(batch_texts): # 리스트가 맞는지, 보낸 텍스트 개수 == 받은 결과 개수
                        for item in data:
                            out_churn_intent.append(item.get("churn_intent"))
                            out_churn_intent_label.append(item.get("churn_intent_label"))
                            out_churn_intent_confidence.append(item.get("churn_intent_confidence"))
                            out_reason.append(item.get("reason"))
                            out_keywords.append(item.get("keywords"))
                        success = True
                        break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(10)
            
            if not success:
                print(f"배치 {i} 최종 실패. 더미 데이터 삽입.")
                out_churn_intent.extend(["Error"] * len(batch_texts))
                out_churn_intent_label.extend([-1] * len(batch_texts))
                out_churn_intent_confidence.extend([0.0] * len(batch_texts))
                out_reason.extend(["Error"] * len(batch_texts))
                out_keywords.extend([[]] * len(batch_texts))

            # 무료 버전일 경우 RPM 5를 넘기면 에러가 발생하므로 지연시간 필요
            print("3초 대기")
            time.sleep(3)

        except Exception as e:
            print(f"Critical Error at batch {i}: {e}")
    end = time.time()
    seconds = int(end-start)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"라벨링완료 - {hours:02d}시간 {minutes:02d}분 {seconds:02d}초 소요")

    # 데이터 저장
    df_sample["churn_intent"] = out_churn_intent
    df_sample["churn_intent_label"] = out_churn_intent_label
    df_sample["churn_intent_confidence"] = out_churn_intent_confidence
    df_sample["reason"] = out_reason
    df_sample["keywords"] = out_keywords

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df_sample.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"저장완료: {args.out} ({len(df_sample)}개)")

    low_confidence_data = df_sample[df_sample['churn_intent_confidence'] < 0.6]
    print(f"검수가 필요한 데이터 개수: {len(low_confidence_data)}")

if __name__ == "__main__":
    main()
