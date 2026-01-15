import re
import argparse
import json
import os
import asyncio
import time
from typing import Any, List, Dict

import pandas as pd
from google import genai  # pip install -U google-genai


# --- 1. 프롬프트 생성 ---

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

### [분류 규칙]
- **비꼬기 주의**: 별점은 낮으나 내용은 긍정적(예: "진짜 빨리 오네요ㅋㅋ")인 경우 문맥상 비꼬는 것(강함)으로 판단하고, 키워드는 실제 의미(예: "배달-지연")로 추출하라.
- **과거 경험 언급**: "저번에도 이러더니 이번에도 그러네요"처럼 반복된 불만이 보이면 즉시 '강함'으로 분류하라.

### [복합 감정 처리 가이드라인]
리뷰에 긍정과 부정이 섞여 있을 경우 아래의 '부정 우선순위'에 따라 라벨링하라.

1. **결정적 부정 (2: 강함)**: 
   - 서비스나 품질에 대한 칭찬이 있더라도 재이용 거부 의사, 치명적 결함, 반복 불만 중 하나라도 포함되는 경우. (예: "다시는 안 먹음", "돈 아깝다", "위생 불량", "태도 불량")
2. **부분적 부정 (1: 약함)**: 
   - 전반적으로 만족하나 특정 부분(배달, 요청사항 미이행, 가격 등)에 명확한 실망을 표현한 경우.
3. **단순 아쉬움 (0: 없음)**: 
   - 만족도가 높으며, 서비스 이탈로 이어질 가능성이 없는 가벼운 피드백.

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
    "churn_intent_reason": "재이용 의사 없음을 명확히 밝히고 배달지연에 대한 문제를 반복적으로 경험하고 있음"
  }}
]
""".strip()


# --- 2. json 내 데이터 추출 ---

def extract_json(s: str) -> Any:
    s = (s or "").strip()
    
    # 마크다운 코드 블록 제거 (```json 또는 ``` 제거)
    s = re.sub(r"```json\s*|```", "", s).strip()
    
    # 대괄호 [ ] 또는 중괄호 { } 사이의 내용만 추출
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


# --- 3. 비동기 처리 ---

async def process_batch(client, model, batch_texts, batch_ratings, batch_index, semaphore) -> List[Dict[str, Any]]:
    async with semaphore: # 동시 요청 수 제한
        prompt = build_batch_prompt(batch_texts, batch_ratings)
        
        for attempt in range(3): # 재시도 로직
            try:
                # 비동기 API 호출
                # 코루틴: 작업단위, 이벤트루프: 스케줄러, asyncio.to_thread: 오래걸리는 동기호출을 스레드로 분리하여 이벤트루프가 막히지 않고 나머지 코루틴 작업들이 계속 진행되게 함
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt,
                    config={"temperature": 0.1},
                )
                
                data = extract_json(resp.text)
                if isinstance(data, list) and len(data) == len(batch_texts):
                    print(f"배치 {batch_index} 완료")
                    out = []
                    for item in data:
                        out.append(
                            {
                                "churn_intent": item.get("churn_intent", ""),
                                "churn_intent_label": item.get("churn_intent_label", -1),
                                "churn_intent_confidence": item.get("churn_intent_confidence", 0.0),
                                "churn_intent_reason": item.get("churn_intent_reason", ""),
                            }
                        )
                    return out
                
            except Exception as e:
                print(f"배치 {batch_index} 시도 {attempt+1} 실패: {e}")
                await asyncio.sleep(2**attempt) # 지수 백오프
        
        print(f"배치 {batch_index} 최종 실패")
        return [{"churn_intent": "Error", "churn_intent_label": -1, "churn_intent_confidence": 0.0, "churn_intent_reason": "Error"} for _ in batch_texts]


# --- 4. main ---

async def main_async():
    p = argparse.ArgumentParser(description="비동기 LLM 이탈의도 라벨링")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--score-col", default="score", help="별점 컬럼명")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--n", type=int, default=200, help="라벨링할 샘플 수")
    p.add_argument("--batch", type=int, default=50, help="한 번에 처리할 샘플 수")
    p.add_argument("--parallel", type=int, default=10, help="동시 실행 배치 수") 
    p.add_argument("--model", default="gemini-2.0-flash", help="Gemini 모델명") # gemini-2.5-flash

    args = p.parse_args()
    
    # 데이터 로드
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.score_col]).copy()
    df = df.head(min(args.n, len(df))).reset_index(drop=True)
    # df_sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    client = genai.Client() # $env:GEMINI_API_KEY='AIzaSy어쩌구'
    semaphore = asyncio.Semaphore(args.parallel)

    tasks = []
    print(f"총 {len(df)}개 데이터를 {args.batch}개씩 비동기 처리 시작")

    start_time = time.time()

    # 배치 단위 태스크 생성
    for i in range(0, len(df), args.batch):
        batch_slice = df.iloc[i : i + args.batch]
        batch_texts = batch_slice[args.text_col].astype(str).tolist()
        batch_ratings = batch_slice[args.score_col].astype(int).tolist()
        tasks.append(process_batch(client, args.model, batch_texts, batch_ratings, i//args.batch + 1, semaphore))

    # 모든 태스크 실행 및 결과 수집
    results = await asyncio.gather(*tasks)
    
    # 컬럼 추가
    out_churn_intent, out_churn_intent_label, out_churn_intent_confidence, out_reason = [], [], [], []
    for batch in results:
        for item in batch:
            out_churn_intent.append(item['churn_intent'])
            out_churn_intent_label.append(item['churn_intent_label'])
            out_churn_intent_confidence.append(item['churn_intent_confidence'])
            out_reason.append(item['churn_intent_reason'])

    df['churn_intent'] = out_churn_intent
    df['churn_intent_label'] = out_churn_intent_label
    df['churn_intent_confidence'] = out_churn_intent_confidence
    df['churn_intent_reason'] = out_reason

    # 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    end_time = time.time()
    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    asyncio.run(main_async())
