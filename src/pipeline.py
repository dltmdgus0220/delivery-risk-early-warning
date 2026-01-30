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


async def main():
    p = argparse.ArgumentParser(description="수집-분류-키워드도출-저장 파이프라인 테스트")
    p.add_argument("--out", required=True, help="결과 CSV 저장 경로")
    p.add_argument("--mode", required=True, choices=["num", "date"], help="수집기준 개수/기간(num/date)")
    p.add_argument("--num", type=int, default=1000, help="수집개수")
    p.add_argument("--start-date", type=str, default="2026-01-01", help="수집시작일")
    p.add_argument("--end-date", type=str, default=None, help="수집종료일")

    args = p.parse_args()
    if args.mode == "num":
        df_test = await run_pipeline(None, collect_mode=args.mode, collect_num=args.num)
    elif args.mode == "date":
        df_test = await run_pipeline(None, collect_mode=args.mode, start_date=args.start_date, end_date=args.end_date)
    df_test.to_csv(args.out, encoding="utf-8-sig", index=False)
    print(f"{len(df_test)}개 저장완료: {args.out}")

if __name__ == "__main__":
    asyncio.run(main())