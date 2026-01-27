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
    combined_inputs = "\n".join([
        f"ID_{i+1}: (별점 {r}점) {t}" for i, (r, t) in enumerate(zip(ratings, texts))
    ])
    
    return f"""
너는 숙련된 고객 경험 분석가이자 데이터 라벨러다. 아래 가이드라인을 바탕으로 배달 앱 리뷰 {len(texts)}개를 분석하라.

### [분류 가이드라인]
- 2 (확정): 명시적 이탈/해지/삭제 선언, 재이용 거부 또는 타서비스(어플) 전환/권유
    - 예: 
        - "다신 안 씀", "탈퇴", "삭제했어요", "삭제할 예정이예요"
        - "이제 안쓸 것 같네요", "쓸 이유가 없네요", "이용못하겠네요", "탈퇴하고 싶어요"
        - "쿠팡으로 갈아탐", "땡겨요 쓰세요", "그냥 전화주문 할게요", "쿠팡이츠쓰는게 나음", "다른 어플쓰세요"

- 1 (불만): 불편/불만/비판/개선 요구 중 하나라도 포함한 경우
    - 예:
        - 부정적 감정 표현: "최악", "짜증", "열받는다", "빡친다", "쓰레기", "소비자기만", "사기꾼", "망해라", "실망", "별로예요"
        - 비꼼 또는 조롱 표현: "음식 다 식은거 먹으란다", "1시간 늦어놓고 3천원 쿠폰 ㅋㅋ", "그렇게 장사하지 마세요^^", "수수료가 4000원 ㅋㅋ"
        - 개선 요청: "너무 좋아요. 근데 수저 포함이 기본으로 설정되면 좋을 것 같아요.", "개선해주세요", "어플이 자꾸 안돼요. 고쳐주세요"
        - 혼합 감정 표현: "다 좋은데 배달이 너무 느려요", "예상시간 자꾸 바뀌는거 빼면 다 좋아요."

- 0 (없음): 순수 긍정 및 만족. **불만사항 또는 건의사항** 없어야 함.
    - 예:
        - "만족해요"
        - "굿"
        - "잘 쓰고 있습니다"
        - "너무 좋아요. 배달의민족 짱"

### [제약 사항]
- 반드시 제공된 ID_1부터 ID_{len(texts)}까지 순서대로 누락 없이 라벨링하라.
- 결과는 반드시 마크다운 없이 JSON 리스트 형식으로만 출력하라.
- 각 리뷰의 'churn_intent_reason' 항목에는 왜 해당 라벨을 선택했는지 가이드라인에 근거하여 짧게 설명하라.
- 입력 개수와 출력 결과의 개수가 반드시 일치해야 한다.

### [입력 데이터]
{combined_inputs}

### [출력 형식 및 예시]
[
  {{
    "id": 1,
    "churn_intent": "확정",
    "churn_intent_label": 2,
    "churn_intent_reason": "재이용 의사 없음을 명확히 밝히고 있음"
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
                    config={"temperature": 0.0},
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
                                "churn_intent_reason": item.get("churn_intent_reason", "")
                            }
                        )
                    return out
                
            except Exception as e:
                print(f"배치 {batch_index} 시도 {attempt+1} 실패: {e}")
                await asyncio.sleep(2**attempt) # 지수 백오프
        
        print(f"배치 {batch_index} 최종 실패")
        return [{"churn_intent": "Error", "churn_intent_label": -1, "churn_intent_reason": "Error"} for _ in batch_texts]


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
    out_churn_intent, out_churn_intent_label, out_churn_intent_reason = [], [], []
    for batch in results:
        for item in batch:
            out_churn_intent.append(item['churn_intent'])
            out_churn_intent_label.append(item['churn_intent_label'])
            out_churn_intent_reason.append(item['churn_intent_reason'])

    df['churn_intent'] = out_churn_intent
    df['churn_intent_label'] = out_churn_intent_label
    df['churn_intent_reason'] = out_churn_intent_reason

    # 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    end_time = time.time()
    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    asyncio.run(main_async())
