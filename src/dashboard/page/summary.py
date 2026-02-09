import json
import sqlite3
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import streamlit as st

from src.dashboard.util import keyword_count, target_keyword_ratio, top_n_keywords_extract, parse_keywords


# --- 유틸 ---
# 한글 폰트 설정
def set_korean_font():
    mpl.rcParams["font.family"] = "NanumGothic"
    # 마이너스 기호 깨짐 방지
    mpl.rcParams["axes.unicode_minus"] = False

# 전역 css
def inject_css():
    st.markdown(
        """
        <style>
          :root{
            --bg:#ffffff;
            --card:#ffffff;
            --muted:#64748b;
            --text:#0f172a;
            --border:#e2e8f0;
            --shadow: 0 1px 2px rgba(15,23,42,.06);
            --radius: 14px;
            --gap: 14px;

            --green:#16a34a;
            --red:#dc2626;
            --blue:#2563eb;
          }

          /* 섹션 카드 */
          .panel{
            background:var(--card);
            border:1px solid var(--border);
            border-radius:var(--radius);
            padding:16px;
            box-shadow:var(--shadow);
          }

          /* KPI 그리드 */
          .kpi-grid{
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--gap);
          }

          /* KPI 카드 */
          .kpi{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 14px 12px;
          }
          .kpi .label{ font-size:13px; color:var(--muted); margin-bottom:6px; }
          .kpi .value{ font-size:28px; font-weight:800; color:var(--text); line-height:1.1; }
          .kpi .sub{ margin-top:6px; font-size:13px; color:var(--muted); }

          /* delta pill */
          .pill{
            display:inline-flex;
            align-items:center;
            gap:6px;
            padding:3px 10px;
            border-radius:999px;
            font-size:12px;
            font-weight:700;
            margin-top:10px;
            border:1px solid transparent;
          }
          .pill.pos{ color:var(--green); background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.18); }
          .pill.neg{ color:var(--red);   background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.18); }

          /* 3개 요약 카드(확정/불만/없음) */
          .mini{
            background:#f8fafc;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
          }
          .mini .title{ font-size:13px; color:var(--muted); margin-bottom:6px; }
          .mini .count{ font-size:20px; font-weight:800; color:var(--text); }
          .mini .ratio{ font-size:12px; color:var(--muted); font-weight:500; margin-left:6px; }

          /* 섹션 타이틀 */
          .section-title{
            font-size:16px;
            font-weight:800;
            margin: 0 0 12px 0;
          }

          /* Streamlit 기본 metric 델타 색이 튀면 숨기고 커스텀으로 통일하고 싶을 때 */
          /* [data-testid="stMetricDelta"] { display:none; } */
        </style>
        """,
        unsafe_allow_html=True,
    )

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
    icon = "▲" if delta_is_good else "▼"

    st.markdown(
        f"""
        <div class="mini">
          <div class="title">{label}</div>
          <div class="count">
            {count:,}건 <span class="ratio">({ratio:.1f}%)</span>
          </div>
          <div class="pill {cls}">{icon} {abs(delta_p):.1f}%p</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 특정 월 데이터 추출
def fetch_month_df(db_path: str, table: str, yyyymm: str) -> pd.DataFrame:
    year, month = map(int, yyyymm.split("-"))

    start_date = date(year, month, 1)
    end_date = start_date + relativedelta(months=1)

    conn = sqlite3.connect(db_path)

    if table == 'data':
        query = f"""
            SELECT *
            FROM {table}
            WHERE at >= ? AND at < ?
        """
        params = (start_date.isoformat(), end_date.isoformat())
    elif table == "summary":
        query = f"""
            SELECT *
            FROM {table}
            WHERE month = ?
        """
        params = (yyyymm,)

    df = pd.read_sql(
        query,
        conn,
        params=params
    )

    conn.close()
    return df

# TopN 키워드 막대그래프 시각화
def render_top_keywords_bar_plotly(df, title: str, top_n=5):
    counter = keyword_count(df)
    top_keywords = top_n_keywords_extract(counter, n=top_n)

    if not top_keywords:
        st.info("표시할 키워드가 없습니다.")
        return None

    chart_df = pd.DataFrame(top_keywords, columns=["keyword", "count"]).sort_values("count")

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
    fig.update_xaxes(title="빈도 수")
    fig.update_yaxes(title=None)

    # 막대 데이터 표시
    fig.update_traces(
        text=chart_df["count"],
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

# 타겟 키워드의 클래스별 비중 비교 시각화
def render_keyword_ratio_compare_bar(
    target: str,
    df_cur_confirmed: pd.DataFrame,
    df_cur_complaint: pd.DataFrame,
    df_prev_confirmed: pd.DataFrame,
    df_prev_complaint: pd.DataFrame,
    cur_label: str = "기준달",
    prev_label: str = "전월",
):
    cur_conf_c = keyword_count(df_cur_confirmed)
    cur_comp_c = keyword_count(df_cur_complaint)
    prev_conf_c = keyword_count(df_prev_confirmed)
    prev_comp_c = keyword_count(df_prev_complaint)

    cur_conf_cnt, cur_conf_ratio = target_keyword_ratio(cur_conf_c, target)
    cur_comp_cnt, cur_comp_ratio = target_keyword_ratio(cur_comp_c, target)
    prev_conf_cnt, prev_conf_ratio = target_keyword_ratio(prev_conf_c, target)
    prev_comp_cnt, prev_comp_ratio = target_keyword_ratio(prev_comp_c, target)

    rows = [
        {"month": cur_label,  "class": "확정", "ratio": cur_conf_ratio, "count": cur_conf_cnt},
        {"month": cur_label,  "class": "불만", "ratio": cur_comp_ratio, "count": cur_comp_cnt},
        {"month": prev_label, "class": "확정", "ratio": prev_conf_ratio, "count": prev_conf_cnt},
        {"month": prev_label, "class": "불만", "ratio": prev_comp_ratio, "count": prev_comp_cnt},
    ]
    plot_df = pd.DataFrame(rows)
    plot_df["label"] = plot_df.apply(lambda r: f"{r['ratio']:.2f}%<br>({r['count']}건)", axis=1)

    max_y = float(plot_df["ratio"].max() if len(plot_df) else 0)
    y_pad = max(1.0, max_y * 0.20) # 위 텍스트 공간

    fig = px.bar(
        plot_df,
        x="class",
        y="ratio",
        color="month",
        barmode="group",
        text="label",
    )

    fig.update_layout(
        height=380, # 왼쪽과 높이 맞춰서 “한 덩어리”로 보이게
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
    )

    fig.update_yaxes(
        title="비율(%)",
        range=[0, max_y + y_pad], # 위 여유
        fixedrange=True,
        showgrid=False,
    )
    fig.update_xaxes(title=None, fixedrange=True)

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# summary 컬럼 추출 (str -> dict)
def _as_dict(x):
    """dict 또는 JSON string을 dict로 변환. 실패하면 빈 dict."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}

# reason_id 컬럼 추출 (str -> list)
def _as_id_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # JSON list 형태면 파싱 시도
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return v
                if isinstance(v, dict) and "reason_id" in v:
                    return _as_id_list(v["reason_id"])
            except Exception:
                pass
    return [x]


# summary
def render_summary_section(title: str, summary_obj):
    d = _as_dict(summary_obj)

    situations = d.get("situations", "")
    evaluations = d.get("evaluations", "")
    solutions = d.get("solutions", "")

    st.markdown(f"**[{title}]**")

    # 보기 좋게 bullet + 본문 분리
    st.markdown("- **문제 상황**")
    st.write(situations if situations else "요약 내용이 없습니다.")

    st.markdown("- **기존 대응에 대한 평가**")
    st.write(evaluations if evaluations else "요약 내용이 없습니다.")

    st.markdown("- **소비자들이 원하는 대응**")
    st.write(solutions if solutions else "요약 내용이 없습니다.")

