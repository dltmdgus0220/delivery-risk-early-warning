import re
import argparse
import json
import os
import time
from typing import Any, List

import pandas as pd
from google import genai  # pip install -U google-genai


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


# --- 5. main ---

def main():
    p = argparse.ArgumentParser(description="LLM(Gemini)로 CSV 일부 자동 라벨링")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--n", type=int, default=1000, help="라벨링할 샘플 수")
    p.add_argument("--batch", type=int, default=100, help="한 번에 처리할 샘플 수")
    p.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini 모델명") # gemini-1.5-flash

    args = p.parse_args()
    
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=args.text_col).copy()
    df_sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    client = genai.Client() # $env:GEMINI_API_KEY='AIzaSy어쩌구'

    out_keywords = []
    batch_size = args.batch # 200까지는 안정적
    
    start = time.time()
    for i in range(0, len(df_sample), batch_size):
        batch_slice = df_sample.iloc[i : i + batch_size]
        batch_texts = batch_slice[args.text_col].astype(str).tolist()

        prompt = build_batch_prompt(batch_texts)
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
                            out_keywords.append(item.get("keywords", []))
                        success = True
                        break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(10)
            
            if not success:
                print(f"배치 {i} 최종 실패")
                out_keywords.extend([[]] * len(batch_texts))

            print("1초 대기")
            time.sleep(1)

        except Exception as e:
            print(f"Critical Error at batch {i}: {e}")
    end = time.time()
    seconds = int(end-start)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"소요 시간: {hours:02d}:{minutes:02d}:{seconds:02d}")

    # 데이터 저장
    df_sample["keywords"] = out_keywords

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df_sample.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"저장완료: {args.out} ({len(df_sample)}개)")

if __name__ == "__main__":
    main()
