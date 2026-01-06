import re
import argparse
import json
import os
import time
from typing import Any, List

import pandas as pd
from google import genai  # pip install -U google-genai


def build_batch_prompt(texts: List[str], task_name: str, labels: List[str], task_desc: str) -> str:
    labels_str = ", ".join([f'"{x}"' for x in labels]) # [없음, 약함, 강함] -> "없음", "약함", "강함", 따옴표로 감싸줌으로써 LLM이 라벨을 맘대로 바꾸는 경우를 방지
    # 텍스트 리스트를 번호가 매겨진 문자열로 변환, 그래야 LLM이 실수없이 잘 이해할 수 있음
    input_texts = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(texts)])
    
    return f"""
너는 숙련된 데이터 라벨러다. 아래 라벨 정의에 따라 {len(texts)}개의 텍스트를 각각 분류한다.
반드시 JSON 리스트만 출력한다(마크다운 금지).
중요: 절대로 생략하지 말고 ID_1부터 ID_50까지 모든 항목에 대해 하나씩 JSON 객체를 생성해라. 답변이 길어져도 끝까지 작성해라.

[태스크]
- task_name: {task_name}
- labels: [{labels_str}]
- description: {task_desc}

[분류 규칙]
1. 반드시 {len(texts)}개의 결과가 포함된 하나의 JSON 리스트만 출력한다.
2. 각 객체는 "label"과 "confidence" 키를 가져야 한다.
3. 텍스트 내용이 짧거나 모호해도 반드시 가장 적절한 라벨을 선택한다.

[입력 텍스트 리스트]
{input_texts}

[출력 JSON 형식(예시)]
[
  {{"label": "없음", "confidence": 0.9}},
  {{"label": "강함", "confidence": 0.8}}
]
""".strip()


def extract_json(s: str) -> Any:
    s = (s or "").strip()
    
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
    p = argparse.ArgumentParser(description="LLM(Gemini)로 CSV 일부 자동 라벨링 (단일 파일 최소 구현)")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--n", type=int, default=200, help="라벨링할 샘플 수")
    p.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini 모델명") # gemini-1.5-flash

    # 태스크(기본: churn_intent)
    p.add_argument("--task-name", default="churn_intent")
    p.add_argument("--labels", default="없음,약함,강함", help="콤마 구분 라벨 목록") # 공백이 있으면 공백단위로 쪼개지기 때문에 쉼표단위로 쪼개기위해 공백은 없어야함.
    p.add_argument(
        "--task-desc",
        default=(
            "리뷰/문장에 '서비스를 떠날 의도(삭제, 탈퇴, 갈아타기, 다시는 안씀 등)'가 있는지 분류. "
            "없음=이탈 의도 없음, 약함=불만은 있으나 이탈이 명시적이지 않음, 강함=이탈/삭제/갈아타기 의도가 명시적임."
        ),
    )

    args = p.parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] # "없음,약함,강함" -> [없음, 약함, 강함]
    
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col]).copy()
    df_sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    client = genai.Client()

    out_labels, out_confs = [], []
    batch_size = 50  # 한 번에 50개씩 묶어서 처리
    
    for i in range(0, len(df_sample), batch_size):
        batch_texts = df_sample[args.text_col].iloc[i : i + batch_size].astype(str).tolist()
        prompt = build_batch_prompt(batch_texts, args.task_name, labels, args.task_desc)
        
        print(f"[{i+1}/{len(df_sample)}] 배치 준비")
        
        try:
            # retry 로직 포함
            success = False
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=args.model, contents=prompt)
                    data = extract_json(resp.text)
                    
                    if isinstance(data, list) and len(data) == len(batch_texts): # 리스트가 맞는지, 보낸 텍스트 개수 == 받은 결과 개수
                        for item in data:
                            out_labels.append(item.get("label"))
                            out_confs.append(item.get("confidence"))
                        success = True
                        break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(20)
            
            if not success:
                # 실패 시 더미 데이터 삽입
                out_labels.extend(["Error"] * len(batch_texts))
                out_confs.extend([0.0] * len(batch_texts))

            # 무료 버전일 경우 RPM 5를 넘기면 에러가 발생하므로 지연시간 필요
            print("20초 대기")
            time.sleep(20)

        except Exception as e:
            print(f"Critical Error at batch {i}: {e}")

    # 데이터 저장
    df_sample[f"{args.task_name}_label"] = out_labels
    df_sample[f"{args.task_name}_confidence"] = out_confs

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True) # 부모디렉토리
    df_sample.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"저장완료: {args.out} ({len(df_sample)}개)")


if __name__ == "__main__":
    main()
