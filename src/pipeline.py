from src.data_collect import collect_reviews_by_num, collect_reviews_by_date
from src.classification.classifier import infer_pipeline
from src.keyword.llm_keyword_async import extract_keywords
from datetime import datetime
import asyncio
import argparse


APP_ID = "com.sampleapp"

async def run_pipeline(conn, collect_mode: str="date", collect_num: int=1000, start_date: str="2026-01-01", end_date: str | None = None):  
    # 데이터 수집
    if collect_mode == "num":
        df = collect_reviews_by_num(APP_ID, collect_num)
    elif collect_mode == "date":
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        df = collect_reviews_by_date(APP_ID, start_dt, end_dt)
    print(f"{len(df)}개 수집완료.")
    # 이탈의도분류
    df = infer_pipeline(df, "model_out/bert-kor-base", text_col="content", batch=16)
    print(f"{len(df)}개 이탈의도분류완료.")
    # 키워드도출
    df = await extract_keywords(df, text_col="content", batch=100)
    print(f"{len(df)}개 키워드도출완료.")

    return df

