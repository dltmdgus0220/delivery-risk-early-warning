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
    
    # 강함 기준에 24년 전후 키워드 조사해서 추가해볼 예정
    return f"""
너는 숙련된 고객 경험 분석가이자 데이터 라벨러다. 아래 가이드라인을 바탕으로 배달 앱 리뷰 {len(texts)}개를 분석하라.

### [분류 가이드라인]
1. **강함 (2)** - 서비스 이탈 가능성 매우 높음
   - 명시적 거부: "다시는 안 씀", "삭제함", "탈퇴함", "돈 아깝다" 등 재이용 거부 의사 표명.
   - 반복적 불만: "저번에도 그러더니 또", "항상 늦음", "벌써 몇 번째인지" 등 동일 문제 반복 언급.
   - 타서비스 표현: "쿠팡이츠가 더 좋네요", "땡겨요 쓰세요" 등 타서비스를 칭찬하고 타서비스 이용을 권장하는 표현 사용.
   
2. **불만 및 건의 (1)** - 불만은 있으나 즉각적인 이탈로 보기 어려운 중간 단계
   - 단순 불만: 일회성 배달 지연, 음식이 다소 식음, 가격이 비쌈 등 일반적인 불편함 토로.
   - 개선 요청/건의: "수저 선택이 기본이었으면 좋겠어요", "가게 숨기기 기능이 있었으면 좋겠어요" 등 더 나은 서비스를 위한 제안.
   - 조건부 불만: "맛은 있는데 배달이 좀 아쉽네요"와 같이 긍정적 요소와 부정적 요소가 섞인 경우.

3. **없음 (0)** - 순수 긍정 및 만족
   - 불만 요소가 전혀 없어야 함.
   - 칭찬, 감사, 만족도 표현만 포함된 경우. (예: "너무 맛있어요", "만족해요", "굿")

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
    "churn_intent_label": 2
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
                                "churn_intent_label": item.get("churn_intent_label", -1)
                            }
                        )
                    return out
                
            except Exception as e:
                print(f"배치 {batch_index} 시도 {attempt+1} 실패: {e}")
                await asyncio.sleep(2**attempt) # 지수 백오프
        
        print(f"배치 {batch_index} 최종 실패")
        return [{"churn_intent": "Error", "churn_intent_label": -1} for _ in batch_texts]


# --- 4. main ---

async def main_async():
    p = argparse.ArgumentParser(description="비동기 LLM 이탈의도 라벨링")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--score-col", default="score", help="별점 컬럼명")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--n", type=int, default=1000, help="라벨링할 샘플 수")
    p.add_argument("--batch", type=int, default=100, help="한 번에 처리할 샘플 수")
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
    out_churn_intent, out_churn_intent_label = [], []
    for batch in results:
        for item in batch:
            out_churn_intent.append(item['churn_intent'])
            out_churn_intent_label.append(item['churn_intent_label'])

    df['churn_intent'] = out_churn_intent
    df['churn_intent_label'] = out_churn_intent_label

    # 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    end_time = time.time()
    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    asyncio.run(main_async())
