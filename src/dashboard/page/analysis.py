import json
import pandas as pd
import plotly.express as px
from collections import Counter
import streamlit as st
from streamlit_plotly_events import plotly_events
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.dashboard.util import fetch_month_df, parse_keywords, set_korean_font, keyword_count, top_n_keywords_extract, detect_keyword_changes


# --- 1. 유틸 ---

# 전역 css
def inject_css():
    st.markdown(
        """
        <style>
          :root{
            --muted:#64748b;
            --text:#0f172a;
            --border:#e2e8f0;
            --green:#16a34a;
            --red:#dc2626;
          }

          /* KPI 카드 */
          .kpi{
            background:#f8fafc;
            border:1px solid var(--border);
            border-radius:12px;
            padding:14px 14px 12px;
          }
          .kpi .label{
            font-size:13px;
            color:var(--muted);
            margin-bottom:6px;
          }
          .kpi .value{
            font-size:28px;
            font-weight:800;
            color:var(--text);
            line-height:1.1;
          }

          /* mini 카드 (확정/불만/없음) */
          .mini{
            background:#f8fafc;
            border:1px solid var(--border);
            border-radius:12px;
            padding:12px;
          }
          .mini .title{
            font-size:13px;
            color:var(--muted);
            margin-bottom:6px;
          }
          .mini .count{
            font-size:20px;
            font-weight:800;
            color:var(--text);
          }
          .mini .ratio{
            font-size:12px;
            color:var(--muted);
            font-weight:500;
            margin-left:6px;
          }

          /* 증감 pill */
          .pill{
            display:inline-flex;
            align-items:center;
            padding:3px 10px;
            border-radius:999px;
            font-size:12px;
            font-weight:700;
            margin-top:10px;
            border:1px solid transparent;
          }
          .pill.pos{
            color:var(--green);
            background:rgba(22,163,74,.10);
            border-color:rgba(22,163,74,.18);
          }
          .pill.neg{
            color:var(--red);
            background:rgba(220,38,38,.10);
            border-color:rgba(220,38,38,.18);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# 3행 키워드 카드 css
def inject_keyword_list_css():
    st.markdown(
        """
        <style>
          .kw-card{
            border:1px solid #e5e7eb;
            border-radius:12px;
            background:#ffffff;
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            overflow:hidden;
          }
          .kw-card-header{
            padding:12px 14px;
            font-weight:800;
            color:#111827;
            font-size:14px;
            background:#ffffff;
            border-bottom:1px solid #eef2f7;
          }
          .kw-row{
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:10px 14px;
            min-height:44px;
            border-bottom:1px solid #eef2f7;
          }
          .kw-row:last-child{ border-bottom:none; }
          .kw-left{
            font-weight:700;
            color:#0f172a;
            font-size:14px;
            max-width:58%;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
          }
          .kw-right{
            display:flex;
            align-items:center;
            gap:10px;
            color:#475569;
            font-size:13px;
            white-space:nowrap;
          }
          .kw-pill{
            min-width:64px;
            height:22px;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            border-radius:999px;
            font-weight:800;
            font-size:12px;
          }
          .kw-pill-empty{
            background:transparent;
            color:transparent;
            border:1px solid transparent;
          }
          .kw-pill-new{
            background:#e0f2fe;
            color:#0369a1;
          }
          .kw-pill-surge{
            background:#fee2e2;
            color:#dc2626;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# DB내 최소, 최대 기간 조회
def get_min_max_yyyymm(db_path: str):
    import sqlite3, pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT MIN(at) AS min_at, MAX(at) AS max_at FROM data", conn)
    conn.close()
    return df.loc[0, "min_at"][:7], df.loc[0, "max_at"][:7]

# 클래스 필터링
def filter_df_by_class(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    if cls == "확정":
        return df[df["churn_intent_label"] == 2].copy()
    if cls == "불만":
        return df[df["churn_intent_label"] == 1].copy()
    return df[df["churn_intent_label"].isin([1, 2])].copy()

# ---- 1행 ----
# 데이터수/이탈지수 카드
def kpi_card(label: str, value: str, delta_text: str, delta_is_good: bool):
    # delta_is_good=True면 초록(긍정), False면 빨강(부정)
    cls = "pos" if delta_is_good else "neg"

    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="pill {cls}">{delta_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

