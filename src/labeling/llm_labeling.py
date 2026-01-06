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

