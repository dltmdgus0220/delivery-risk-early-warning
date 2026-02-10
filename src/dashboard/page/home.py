import sqlite3
import asyncio
import altair as alt
import pandas as pd
import streamlit as st
from datetime import date, datetime, timedelta
from src.dashboard.pipeline import run_pipeline
from src.dashboard.util import fetch_period_df, set_korean_font, parse_keywords, keyword_count, top_n_keywords_extract
from src.risk_summary.risk_score_calc import risk_score_calc


# --- 1. 유틸 ---

DATE_COL = "at"
DATA_TABLE = "data"
SUMMARY_TABLE = "summary"

# 데이터 포멧
def _to_date(x):
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])

def _fmt_yy_mm_dd(s: str) -> str:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.strftime("%y.%m.%d")

def _fmt_k(n: int) -> str:
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return f"{n}"

# 1달전/2달전 날짜 리턴
def _month_range(today:datetime, offset_months: int = 0):
    """offset_months=0 1달전, -1 2달전 (today 기준)"""
    first_month = today.replace(day=1) - timedelta(days=1)

    y = first_month.year
    m = first_month.month + offset_months
    while m <= 0:
        y -= 1
        m += 12
    while m >= 13:
        y += 1
        m -= 12

    start = datetime(y, m, 1)
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = next_month - timedelta(days=1)
    return start, end

# DB내 저장된 기간 및 데이터수 조회
def _minmax_and_total(conn):
    cur = conn.execute(
        f"""
        SELECT MIN(date({DATE_COL})), MAX(date({DATE_COL})), COUNT(*)
        FROM data
        """
    )
    mn, mx, total = cur.fetchone()
    cur.close()
    return mn, mx, int(total)


        f"""
        """,
    )

    """
    """

    )
    )




def render_sidebar(today):
    st.sidebar.subheader("🔄 데이터 관리")

    db_path = st.sidebar.text_input(
        "DB 경로",
        value="demo.db"
    )

    if st.sidebar.button("데이터 갱신", use_container_width=True):
        status = st.sidebar.empty()
        status.info("파이프라인 실행 중...")

        try:
            conn = sqlite3.connect(db_path)
            flag = asyncio.run(run_pipeline(conn, today))
            conn.close()
            status.empty()

            if flag == 0:
                st.sidebar.success("데이터 갱신 완료!")
            else:
                st.sidebar.success("이미 최신 데이터입니다.")

        except Exception as e:
            status.empty()
            st.sidebar.error(f"실행 실패: {e}")

    st.sidebar.divider()
    st.sidebar.subheader("DB 요약")

    try:
        conn = sqlite3.connect(db_path)

        cur_s, cur_e = _month_range(today, 0)      # 이번 달
        prev_s, prev_e = _month_range(today, -1)  # 지난 달


        mn, mx, total = _minmax_and_total(conn)
        conn.close()

        line1 = (
            f"이번 달 데이터 : "
            f"{_fmt_yy_mm_dd(cur_s.strftime('%Y-%m-%d'))}"
            f"~{_fmt_yy_mm_dd(cur_e.strftime('%Y-%m-%d'))} "
            f"(총 {cur_cnt}개)"
        )
        line2 = (
            f"지난 달 데이터 : "
            f"{_fmt_yy_mm_dd(prev_s.strftime('%Y-%m-%d'))}"
            f"~{_fmt_yy_mm_dd(prev_e.strftime('%Y-%m-%d'))} "
            f"(총 {prev_cnt}개)"
        )
        line3 = (
            f"전체 데이터 : "
            f"{_fmt_yy_mm_dd(mn)}~{_fmt_yy_mm_dd(mx)} "
            f"(총 {_fmt_k(total)}개)"
        )

        st.sidebar.text(line1 + "\n" + line2 + "\n" + line3)

    except Exception as e:
        st.sidebar.caption(f"DB 요약을 불러오지 못했습니다: {e}")
    return {
    }


def render(cfg: dict, today: datetime):









    # 클래스별 키워드 TopN
    st.divider()
    st.subheader("클래스별 키워드 TopN")

    top_n = st.slider("Top N", 5, 30, 10, 1)


    buckets = ["확정", "불만", "없음"]
    cols = st.columns(3, gap="large")

    for i, label in enumerate(buckets):
        with cols[i]:
            st.markdown(f"#### '{label}' 키워드 Top{top_n}")

                st.caption("데이터가 없습니다.")
            else:
                bar = (
                    .mark_bar()
                    .encode(
                    )
                )
