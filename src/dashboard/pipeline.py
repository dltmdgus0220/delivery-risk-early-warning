# 솔루션 반영해서 넣어야함.
import json
import pandas as pd
import sqlite3
from src.data_collect import collect_reviews_by_date
from src.classification.classifier import infer_pipeline
from src.keyword.llm_keyword_async import extract_keywords
from src.risk_summary.risk_score_calc import risk_score_calc
from src.risk_summary.llm_summary_reviews import summary_pipeline
from src.dashboard.util import fetch_month_df, delete_month_df
from datetime import datetime, timedelta
import asyncio
import argparse


# --- 1. 상수 선언 및 기타 함수 ---
DB_PATH = "demo.db"
APP_ID = "com.sampleapp"
DATE_COL = "at"
TODAY = datetime(2026, 2, 1)

# 안전하게 리스트를 문자열로 변환
def safe_json_dumps(x):
    if isinstance(x, list): # 리스트 -> 문자열
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, str): # 이미 문자열이면 그대로
        return x
    return json.dumps([], ensure_ascii=False) # 나머지는 빈 리스트로  

# SQLiteDB 적재
def save_db(df:pd.DataFrame, conn, table:str, if_exists:str="append", chunksize:int=5000):
    df.to_sql(
        name=table,
        con=conn,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",  # 여러 row를 한 번에 insert
    )
    conn.commit()
    
    # 저장 확인
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    total_rows = cur.fetchone()[0]

    cur.execute(f"PRAGMA table_info({table})")
    col_info = cur.fetchall()
    columns = [c[1] for c in col_info]

    if table == "data":
        cur.execute(f"SELECT * FROM {table} LIMIT 5")
    elif table == "summary":
        cur.execute(f"SELECT * FROM {table} ORDER BY month DESC LIMIT 5")
    sample_rows = cur.fetchall()

    sample_df = pd.DataFrame(sample_rows, columns=columns)

    print(f"DB 적재 완료.")
    print("테이블:", table)
    print("이번에 저장한 행 수:", len(df))
    print("테이블 전체 행 수:", total_rows)
    print("컬럼:", columns)
    print("\n샘플 5행:")
    print(sample_df)


# --- 2. 파이프라인 ---

async def run_pipeline(conn, today, data_table:str="data", summary_table:str="summary", if_exists:str="append", chunksize:int=5000):
    # 매달 1일에 지난 달 리뷰 보는 상황 가정
    # 날짜 계산
    # start_date = today.replace(day=1) # 이번 달 1일로 변경
    # end_date = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1) # 이번 달 말일
    end_date = (today.replace(day=1) - timedelta(days=1)) # 지난 달 말일
    start_date = end_date.replace(day=1) # 지난 달 1일
    yyyymm = start_date.strftime("%Y-%m")

    # 이번 달 데이터 있는지 확인
    # df_tmp = fetch_month_df(DB_PATH, data_table, yyyymm)
    df_cur = collect_reviews_by_date(APP_ID, start_date, end_date)
    
    # 이번 달 데이터 추가 수집 -> DB 적재
    # if len(df_tmp) != len(df_cur): # 중복 처리 조건 변경필요
    # 재현성을 위해 삭제하고 다시 수집
    delete_month_df(DB_PATH, data_table, yyyymm)
    delete_month_df(DB_PATH, summary_table, yyyymm)
    print(f"{len(df_cur)}개 수집 완료. ({start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')})")
    
    # 이탈의도분류
    df_cur = infer_pipeline(df_cur, "model_out/bert-kor-base", text_col="content", batch=16)
    df_cur0 = df_cur[df_cur['churn_intent_label'] == 0].copy()
    df_cur1 = df_cur[df_cur['churn_intent_label'] == 1].copy()
    df_cur2 = df_cur[df_cur['churn_intent_label'] == 2].copy()
    print(f"{len(df_cur)}개 이탈의도 분류 완료. (확정:{len(df_cur2)}개/불만:{len(df_cur1)}개/없음:{len(df_cur0)}개)")
    
    # 키워드도출
    df_cur = await extract_keywords(df_cur, text_col="content", batch=100)
    print(f"{len(df_cur)}개 키워드 도출 완료.")
    # 키워드 변환
    df_cur["keywords"] = df_cur["keywords"].map(safe_json_dumps)
    # df_cur.to_csv('out_test1.csv', encoding='utf-8-sig')

    # 데이터 DB 적재
    save_db(df_cur, conn, data_table, if_exists, chunksize)

    # 이탈지수계산
    risk_score = risk_score_calc(df_cur)
    print("이탈 지수 계산 완료")
    # '확정' 키워드 기반 리뷰 요약
    summary_complaint, summary_confirmed, target = summary_pipeline(df_cur)
    print("요약 카드 생성 완료")

    df_summary = pd.DataFrame([{
        "month": start_date.strftime('%Y-%m'),
        "risk_score": risk_score,
        "target": target,
        "summary_complaint": summary_complaint,
        "summary_confirmed": summary_confirmed,
        "solution": "대안없음"
        }])

    for col in ["summary_complaint", "summary_confirmed"]:
        df_summary[col] = df_summary[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)

    # 요약 DB 적재
    save_db(df_summary, conn, summary_table, if_exists, chunksize)
    return 0

    # else:
    #     print("이미 최신 데이터 입니다.")
    #     return 1


async def main():
    p = argparse.ArgumentParser(description="수집-분류-키워드도출-저장-요약 파이프라인")
    p.add_argument("--db-path", required=True, help="DB 저장 경로")
    p.add_argument("--data-table", type=str, default="data", help="데이터 저장할 테이블 이름")
    p.add_argument("--summary-table", type=str, default="summary", help="리뷰 요약 저장할 테이블 이름")
    p.add_argument("--if-exists", type=str, default="append", choices=["append", "replace"], help="이미 테이블이 존재하는 경우 어떻게 저장할건지")
    p.add_argument("--chunksize", type=int, default=5000, help="한 번에 DB에 적재할 청크사이즈")

    args = p.parse_args()
    
    conn = sqlite3.connect(args.db_path)
    try:
        await run_pipeline(conn, TODAY, args.data_table, args.summary_table, args.if_exists, args.chunksize)
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(main())