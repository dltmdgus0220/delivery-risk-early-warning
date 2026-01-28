import re
import argparse
import json
import os
import asyncio
import time
from typing import Any, List

import pandas as pd
from google import genai


# --- 1. 사전 정의 ---

ASPECT = ["배달", "배달원", "배차", "배달지역", "배달수수료", "최소주문금액", "배달예상시간",
          "고객센터", "보상", "대응",
          "쿠폰", "혜택", "이벤트", "멤버십", "광고", "구독료",
          "결제", "주문", "리뷰", "음식상태", "위생",
          "로그인", "본인인증", "정책", "기업",
          "앱", "업데이트", "UI", "UX", "기능"]

STATUS = ["지연", "느림", "빠름",
          "최악", "불만", "만족", "나쁨", "좋음",
          "많음", "적음", "없음", "미흡",
          "친절", "불친절", "편함", "불편",
          "불가", "오류", "정확", "취소", 
          "비쌈", "저렴", "삭제", "탈퇴", "거부",
          "건의", "개선요청"]

# 배민 고유 서비스
SERVICE_KEYWORDS = ["한집배달", "가게배달", "알뜰배달", "배민1", "배민스토어", "B마트", "배민클럽", "배민패스"]


# --- 2. 프롬프트 생성 ---

def build_batch_prompt(texts: List[str]) -> str:
    # LLM이 실수없이 잘 이해할 수 있도록 "ID_1: 맛있어요" 형식으로 묶음.
    text_inputs = "\n".join([
        f"ID_{i+1}: {t}" for i, t in enumerate(texts)
    ])
    
    return f"""
너는 숙련된 고객 경험 분석가이자 데이터 라벨러다. 아래 가이드라인을 바탕으로 배달 앱 리뷰 {len(texts)}개를 분석하라.

### [STATUS 리스트 및 키워드 규칙]
1. **키워드 형식**: 반드시 `[대상]-[상태]` 형태의 짧은 단어로 추출하라.
2. **고유 서비스명 보존**: 리뷰 내용 중 **{SERVICE_KEYWORDS}**와 관련된 언급이 있다면, 키워드 추출 시 일반적인 '배달' 대신 해당 고유 명칭을 반드시 사용하라.
3. **대상값(Aspect) 사용 제한 (매우 중요)**: 하이픈(-) 앞의 [대상] 값은 **반드시** {ASPECT} 또는 {SERVICE_KEYWORDS} 리스트에 정의된 단어만 사용 가능하다.
4. **상태값(Status) 사용 제한 (매우 중요)**: 하이픈(-) 뒤의 [상태] 값은 **반드시** {STATUS} 리스트에 정의된 단어만 사용 가능하다. 
5. **강제 매핑 규칙**: 만약 리뷰 내용에 적합한 단어가 리스트에 없다면, 리스트 내에서 **의미가 가장 유사한 단어**를 선택하라. 
   - 예: "음식이 식어서 왔어요" -> "음식상태-나쁨" (리스트에 '식음'이 없으므로)
   - 예: "맛있어요" -> "음식상태-좋음" (리스트에 '맛있음'이 없으므로)
   - 예: "유튜브 프리미엄 연동이 안되요" -> "기능-불가"
   - 예: "앱 열때마다 회원가입 권유하네요. 지긋지긋합니다" -> "UX-불만"
   - 예: "수저 기본으로 설정해줬으면 좋겠어요" -> "UX-개선요청"
   - 예: "이제 회원탈퇴하고 안쓸게요" -> "앱-탈퇴"
5. **리스트 외 단어 사용 금지**: 리스트에 없는 단어(예: 느려요, 맛없음, 별로임 등)를 사용할 경우 해당 결과는 오류로 처리된다.
6. **복합 감정 처리**: 복합 감정인 경우, 만족한 부분과 불만족한 부분을 모두 키워드로 추출하라. 
   - 예: "맛은 있는데 너무 늦어요" -> keywords: ["음식상태-좋음", "배달-지연"]

### [제약 사항]
- 반드시 제공된 ID_1부터 ID_{len(texts)}까지 순서대로 누락 없이 라벨링하라.
- 결과는 반드시 마크다운 없이 JSON 리스트 형식으로만 출력하라.
- 입력 개수와 출력 결과의 개수가 반드시 일치해야 한다.
- 출력 직전, 모든 `keywords`의 하이픈 앞 단어가 **{ASPECT} 또는 {SERVICE_KEYWORDS}** 리스트에 포함되어 있는지 최종 검토하라.
- 출력 직전, 모든 `keywords`의 하이픈 뒤 단어가 **{STATUS}** 리스트에 포함되어 있는지 최종 검토하라.

### [입력 데이터]
{text_inputs}

### [출력 형식 및 예시]
[
  {{
    "id": 1,
    "keywords": ["배달-지연", "대응-미흡", "보상-없음"]
  }}
]
""".strip()


# --- 3. json 추출 ---

def extract_json(s: str) -> Any:
    s = (s or "").strip()
    
    # 마크다운 코드 블록 제거 (```json 또는 ``` 제거)
    s = re.sub(r"```json\s*|```", "", s).strip()
    
    # 대괄호 [ ] 또는 중괄호 { } 사이의 내용만 추출
    # 배치 처리 시에는 리스트([])로 올 확률이 높으므로 둘 다 대응
    start_idx = s.find("[")
    end_idx = s.rfind("]")
    
    if start_idx == -1 or end_idx == -1:
        start_idx = s.find("{")
        end_idx = s.rfind("}")

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(f"유효한 JSON 형식을 찾을 수 없습니다. 응답 내용: {s[:100]}...")
        
    json_str = s[start_idx : end_idx + 1]
    return json.loads(json_str)


# --- 4. 비동기 처리 ---

async def process_batch(client, model, batch_texts, batch_index, semaphore) -> List[List[str]]:
    async with semaphore: # 동시 요청 수 제한
        prompt = build_batch_prompt(batch_texts)
        
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
                    return [item.get("keywords", []) for item in data]
                
            except Exception as e:
                print(f"배치 {batch_index} 시도 {attempt+1} 실패: {e}")
                await asyncio.sleep(2**attempt) # 지수 백오프
        
        print(f"배치 {batch_index} 최종 실패")
        return [[] for _ in batch_texts]


# --- 5. main ---

async def main_async():
    p = argparse.ArgumentParser(description="비동기 LLM 키워드도출")
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", default="content")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--parallel", type=int, default=10) # 동시 실행 배치 수
    p.add_argument("--model", default="gemini-2.0-flash")

    args = p.parse_args()
    
    # 데이터 로드
    df = pd.read_csv(args.csv)
    df = df.head(min(args.n, len(df))).reset_index(drop=True)
    # df_sample = df.sample(n=min(args.n, len(df)), random_state=42).reset_index(drop=True)

    client = genai.Client()
    semaphore = asyncio.Semaphore(args.parallel)
    
    tasks = []
    print(f"총 {len(df)}개 데이터를 {args.batch}개씩 비동기 처리 시작")
    
    start_time = time.time()

    # 배치 단위 태스크 생성
    for i in range(0, len(df), args.batch):
        batch_texts = df.iloc[i : i + args.batch][args.text_col].astype(str).tolist()
        tasks.append(process_batch(client, args.model, batch_texts, i//args.batch + 1, semaphore))

    # 모든 태스크 실행 및 결과 수집
    results = await asyncio.gather(*tasks)
    
    # 2차원 리스트 평탄화
    flat_results = [item for batch in results for item in batch]
    df["keywords"] = flat_results

    # 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    
    end_time = time.time()
    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print(f"저장 완료: {args.out}")

if __name__ == "__main__":
    asyncio.run(main_async())