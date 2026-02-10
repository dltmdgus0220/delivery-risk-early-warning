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

# 클래스별 변화 카드
def class_mini_card(label, count, ratio, delta_p, delta_is_good: bool):
    # delta_p가 +면 좋다/나쁘다는 정책이 있을 텐데, 지금은 "증가=초록"으로 유지
    cls = "pos" if delta_is_good else "neg"

    st.markdown(
        f"""
        <div class="mini">
          <div class="title">{label}</div>
          <div class="count">
            {count:,}건 <span class="ratio">({ratio:.1f}%)</span>
          </div>
          <div class="pill {cls}"> {(delta_p):.1f}%p</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- 2행 ----
# TopN 가로막대그래프 시각화
def render_top_keywords_bar_plotly(df, title: str, top_n=5):
    counter = keyword_count(df)
    top_keywords = top_n_keywords_extract(counter, n=top_n)

    if not top_keywords:
        st.info("표시할 키워드가 없습니다.")
        return None

    chart_df = pd.DataFrame(top_keywords, columns=["keyword", "count"]).sort_values("count")

    # 비율 계산 (전체 키워드 등장 횟수 기준)
    total = sum(counter.values())
    if total == 0:
        chart_df["ratio"] = 0.0
    else:
        chart_df["ratio"] = (chart_df["count"] / total) * 100

    # 막대 끝 라벨: "00건 (00.0%)"
    chart_df["label"] = chart_df.apply(
        lambda r: f"{int(r['count'])}건<br>({r['ratio']:.1f}%)",
        axis=1,
    )

    max_x = int(chart_df["count"].max()) if len(chart_df) else 0
    pad = max(1, int(max_x * 0.14))

    fig = px.bar(
        chart_df,
        x="count",
        y="keyword",
        orientation="h",
        title=title,
    )

    # 타이틀
    fig.update_layout(
    title=dict(
        text=title,
        x=0.5, # 중앙 정렬
        xanchor="center",
        font=dict(size=20, family="Arial", color="black"),
    ),
    margin=dict(l=10, r=10, t=50, b=10),
    )

    # 축 이름설정
    fig.update_xaxes(title="빈도 수", range=[0, max_x + pad])
    fig.update_yaxes(title=None)

    # 막대 데이터 표시
    fig.update_traces(
        text=chart_df["label"],
        textposition="outside",
    )

    fig.update_layout(clickmode="event+select")
    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        key="top_keyword_bar",
        on_select="rerun",
    )

    if selected['selection']['points'] != []:
        return top_keywords, selected['selection']['points'][0]['y']  # 클릭한 키워드

    return top_keywords, None

# 키워드 추이 시각화
# month 리스트 생성
def build_11mo_window(center_yyyymm: str, min_yyyymm: str | None = None, max_yyyymm: str | None = None):
    """
    center_yyyymm을 중앙으로 11개월 리스트 생성.
    - 과거가 부족하면 미래로 보충
    - 미래가 부족하면 과거로 보충
    min_yyyymm/max_yyyymm은 "YYYY-MM" 형식(데이터 존재 가능한 범위)
    """
    center_dt = datetime.strptime(center_yyyymm, "%Y-%m")

    start_dt = center_dt - relativedelta(months=5)
    end_dt = center_dt + relativedelta(months=5)

    min_dt = datetime.strptime(min_yyyymm, "%Y-%m") if min_yyyymm else None
    max_dt = datetime.strptime(max_yyyymm, "%Y-%m") if max_yyyymm else None

    # 1) 과거 경계 보정: start가 min보다 앞이면 부족분만큼 end를 뒤로 밀기
    if min_dt and start_dt < min_dt:
        diff = (min_dt.year - start_dt.year) * 12 + (min_dt.month - start_dt.month)  # 부족 개월 수
        start_dt = min_dt
        end_dt = end_dt + relativedelta(months=diff)

    # 2) 미래 경계 보정: end가 max보다 뒤면 부족분만큼 start를 앞으로 밀기
    if max_dt and end_dt > max_dt:
        diff = (end_dt.year - max_dt.year) * 12 + (end_dt.month - max_dt.month)  # 초과 개월 수
        end_dt = max_dt
        start_dt = start_dt - relativedelta(months=diff)

        # 2-1) start를 앞으로 밀었더니 min보다 더 앞서면 다시 min으로 고정
        if min_dt and start_dt < min_dt:
            start_dt = min_dt

    # 3) 최종 months 만들기 (start~end 범위에서 최대 11개)
    months = []
    cur = start_dt
    while cur <= end_dt and len(months) < 11:
        months.append(cur.strftime("%Y-%m"))
        cur = cur + relativedelta(months=1)

    return months
