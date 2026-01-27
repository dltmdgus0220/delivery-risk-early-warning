import re
import argparse
import json
import os
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
너는 배달 앱 리뷰의 이탈 의도를 분석하는 숙련된 CX 전략가다. 
아래 가이드라인과 5단계 분류 절차를 엄격히 준수하여 데이터를 라벨링하라.

### [분류 가이드라인]
- **2 (확정):** 이탈 선언, 앱 삭제/탈퇴, 재이용 거부, 타사 서비스로의 전환 및 추천.
- **1 (불만):** 단순 불만, 기능적 결함, 개선 요구, 비꼼, 조롱, 혹은 칭찬 뒤에 붙은 단점 언급.
- **0 (없음):** 불만 요소가 전혀 없는 순수 만족 및 긍정적 피드백.

### [5단계 분류 절차 - 순차 적용]
1. **(이탈 확정):** 삭제, 탈퇴, 안씀, 다신 안시킴 등 명시적 거부 표현이 있는가? -> YES: **2**
2. **(타사 전환):** 타 서비스(쿠팡, 요기요 등)로 갈아탄다거나 타 서비스를 권유하는가? -> YES: **2**
3. **(건의/개선):** "개선해주세요", "비싸요", "늦어요" 등 불편함이나 요청사항이 있는가? -> YES: **1**
4. **(비꼼/부정):** 낮은 별점과 함께 비꼬는 말투(예: "참 빨리 오네요^^")나 비난이 있는가? -> YES: **1**
5. **(순수 긍정):** 위 모든 조건에 해당하지 않는 깨끗한 만족 리뷰인가? -> YES: **0** (그 외 판단 불능 시에도 0 부여)

### [제약 사항]
- 결과는 반드시 **마크다운 없이 순수 JSON 리스트 형식**으로만 출력하라. (예: `[...]`)
- 입력 개수와 출력 결과의 개수(ID_{len(texts)}까지)가 반드시 일치해야 한다.
- `churn_intent_reason`은 'X단계 기준 충족'이라는 내용을 포함하여 간결하게 작성하라.

### [입력 데이터]
{combined_inputs}

### [출력 형식]
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

    start_idx = s.find("[")
    end_idx = s.rfind("]")

    if start_idx == -1 or end_idx == -1:
        start_idx = s.find("{")
        end_idx = s.rfind("}")

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(f"유효한 JSON 형식을 찾을 수 없습니다. 응답 내용: {s[:100]}...")

    json_str = s[start_idx: end_idx + 1]
    return json.loads(json_str)


# --- 3. 배치 단위 처리 ---

def process_batch(client, model, batch_texts, batch_ratings, batch_index) -> List[Dict[str, Any]]:
    prompt = build_batch_prompt(batch_texts, batch_ratings)

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
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
            time.sleep(2 ** attempt)

    print(f"배치 {batch_index} 최종 실패")
    return [{"churn_intent": "Error", "churn_intent_label": -1, "churn_intent_reason": "Error"} for _ in batch_texts]


def label_subset(df_sub: pd.DataFrame, client, args) -> pd.DataFrame:
    if df_sub.empty:
        return df_sub.copy()

    out_churn_intent, out_churn_intent_label, out_churn_intent_reason = [], [], []

    total = len(df_sub)
    batch_no = 0

    for i in range(0, total, args.batch):
        batch_no += 1
        batch_slice = df_sub.iloc[i: i + args.batch]
        batch_texts = batch_slice[args.text_col].astype(str).tolist()
        batch_ratings = batch_slice[args.score_col].astype(int).tolist()

        batch_out = process_batch(
            client=client,
            model=args.model,
            batch_texts=batch_texts,
            batch_ratings=batch_ratings,
            batch_index=batch_no
        )

        for item in batch_out:
            out_churn_intent.append(item["churn_intent"])
            out_churn_intent_label.append(item["churn_intent_label"])
            out_churn_intent_reason.append(item["churn_intent_reason"])

    df_out = df_sub.copy()
    df_out["churn_intent"] = out_churn_intent
    df_out["churn_intent_label"] = out_churn_intent_label
    df_out["churn_intent_reason"] = out_churn_intent_reason
    return df_out


# --- 4. main ---

def main():
    p = argparse.ArgumentParser(description="동기 LLM 이탈의도 라벨링")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--score-col", default="score", help="별점 컬럼명")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--n", type=int, default=1000, help="라벨링할 샘플 수")
    p.add_argument("--batch", type=int, default=100, help="한 번에 처리할 샘플 수")
    p.add_argument("--model", default="gemini-2.0-flash", help="Gemini 모델명")
    p.add_argument("--rerun-max", type=int, default=2,
                   help="라벨링 후 churn_intent가 ['확정','불만','없음']에 없는 행만 재라벨링 반복 횟수")

    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.score_col]).copy()
    df = df.head(min(args.n, len(df))).reset_index(drop=True)

    client = genai.Client()  # $env:GEMINI_API_KEY='...'

    start_time = time.time()
    print(f"총 {len(df)}개 데이터를 {args.batch}개씩 동기 처리 시작")

    # 1차 라벨링(전체)
    df_labeled = label_subset(df, client, args)

    # 라벨 이상치 행만 재라벨링 반복
    VALID = ["확정", "불만", "없음"]

    for rerun in range(1, args.rerun_max + 1):
        bad_mask = ~df_labeled["churn_intent"].isin(VALID)
        bad_cnt = int(bad_mask.sum())

        print(f"[재라벨링 {rerun}/{args.rerun_max}] churn_intent 이상치 {bad_cnt}건")
        if bad_cnt == 0:
            break

        df_bad = df_labeled.loc[bad_mask].copy()
        df_bad_labeled = label_subset(df_bad, client, args)

        df_labeled.loc[df_bad_labeled.index, "churn_intent"] = df_bad_labeled["churn_intent"]
        df_labeled.loc[df_bad_labeled.index, "churn_intent_label"] = df_bad_labeled["churn_intent_label"]
        df_labeled.loc[df_bad_labeled.index, "churn_intent_reason"] = df_bad_labeled["churn_intent_reason"]

    bad_mask = ~df_labeled["churn_intent"].isin(VALID)
    if bad_mask.any():
        print(f"최종적으로도 이상치 {int(bad_mask.sum())}건 남음 (결과 그대로 저장)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_labeled.to_csv(args.out, index=False, encoding="utf-8-sig")

    end_time = time.time()
    print(f"소요 시간: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
