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

# 월별 키워드 비율 계산
def monthly_keyword_ratio(
    db_path: str,
    months: list[str],
    keyword: str,
    cls: str,
):
    """
    months: ["2025-08", ..., "2026-06"] 같은 11개월
    cls: "확정"|"불만"|"확정+불만"
    """
    rows = []

    for yyyymm in months:
        df_m = fetch_month_df(db_path, "data", yyyymm)
        if len(df_m) == 0:
            rows.append({"yyyymm": yyyymm, "ratio": 0.0, "count": 0, "total": 0})
            continue

        df_m["keywords"] = df_m["keywords"].apply(parse_keywords)

        # 클래스 필터
        if cls == "확정":
            df_m = df_m[df_m["churn_intent_label"] == 2]
        elif cls == "불만":
            df_m = df_m[df_m["churn_intent_label"] == 1]
        else:  # 확정+불만
            df_m = df_m[df_m["churn_intent_label"].isin([1, 2])]

        counter = keyword_count(df_m)
        total = sum(counter.values())
        count = counter.get(keyword, 0)
        ratio = 0.0 if total == 0 else round(count / total * 100, 2)

        rows.append({"yyyymm": yyyymm, "ratio": ratio, "count": count, "total": total})

    return pd.DataFrame(rows)

# 키워드 추이 꺾은선그래프 시각화
def render_keyword_trend_line(df_trend: pd.DataFrame, title: str, center_yyyymm: str):
    # 가운데 기준달 표시용(세로선)
    fig = px.line(
        df_trend,
        x="yyyymm",
        y="ratio",
        markers=True,
        text=df_trend["ratio"].map(lambda x: f"{x:.1f}%"),
    )

    fig.update_traces(textposition="top center")

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=20,
                family="Arial",
                color="black",
            ),
        ),
        height=450,
        margin=dict(l=10, r=30, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(title=None, showgrid=False)
    fig.update_yaxes(title="비중(%)", showgrid=False, zeroline=False)

    # 기준달(중앙) vertical line
    fig.add_vline(
        x=center_yyyymm,
        line_width=2,
        line_dash="dash",
        line_color="rgba(37,99,235,0.6)",
    )

    st.plotly_chart(fig, use_container_width=True)

# ---- 3행 ----
# 신규+급증 렌더링
def render_keyword_list_card(
    title: str,
    rows: list[dict],
    top_k: int,
    mode: str,  # "new" | "surge"
):
    """
    rows:
      - new:   {"keyword", "cur_count", "cur_ratio"}
      - surge: {"keyword", "cur_count", "cur_ratio", "diff_pp"}
    """
    st.markdown(f'<div class="kw-card"><div class="kw-card-header">{title}</div>', unsafe_allow_html=True)

    if not rows:
        st.markdown(
            '<div style="padding:12px 14px; color:#64748b; font-size:13px;">표시할 항목이 없습니다.</div></div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(rows).head(top_k).copy()

    # 공통: 현재 비중(%)
    df["cur_ratio_pct"] = (df["cur_ratio"] * 100).round(1)
    df["cur_count"] = df["cur_count"].astype(int)

    # surge만: 증가폭(%p)
    if mode == "surge":
        df["diff_pp_pct"] = (df["diff_pp"] * 100).round(1)

    for _, r in df.iterrows():
        keyword = r["keyword"]
        right_text = f"{r['cur_count']}건 | {r['cur_ratio_pct']}%"

        if mode == "new":
            pill_cls = "kw-pill-new"
            pill_text = "NEW"
        else:
            pill_cls = "kw-pill-surge"
            pill_text = f"+{r['diff_pp_pct']}%p"

        st.markdown(
            f"""
            <div class="kw-row">
              <div class="kw-left" title="{keyword}">{keyword}</div>
              <div class="kw-right">
                <span>{right_text}</span>
                <span class="kw-pill {pill_cls}">{pill_text}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# 키워드 검색 헬퍼
def top_keywords_for_suggest(df_cls: pd.DataFrame, top_k: int = 20):
    c = keyword_count(df_cls)
    top = top_n_keywords_extract(c, n=top_k)
    return [k for k, _ in top], c

# 동시발생 키워드 조회
def cooccur_top(
    df_cls: pd.DataFrame,
    target_kw: str,
    top_k: int = 10,
):
    """
    target_kw와 같은 리뷰에서 같이 등장한 키워드 TopK 반환.
    반환: list[dict] = [{"keyword":..., "count":..., "ratio":...}]
    ratio는 (target_kw 포함 리뷰 중 해당 키워드 동시발생 비율) 기준
    """
    if df_cls.empty:
        return []

    # target 포함 리뷰만
    mask = df_cls["keywords"].apply(lambda ks: target_kw in ks)
    df_t = df_cls[mask].copy()
    base = len(df_t)
    if base == 0:
        return []

    co = Counter()
    for ks in df_t["keywords"]:
        # 같은 리뷰에서 target 제외하고 카운트
        for k in ks:
            if k != target_kw:
                co[k] += 1

    # TopK
    top = co.most_common(top_k)

    out = []
    for k, cnt in top:
        ratio = round(cnt / base * 100, 1)  # 기준: target 포함 리뷰 중 비율
        out.append({"keyword": k, "count": cnt, "ratio": ratio})

    return out, base

# 동시발생 키워드 렌더링
def card_container(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div class="kw-card">
          <div class="kw-card-header">
            {title}
            {f"<div style='font-size:12px;color:#64748b;margin-top:4px;'>{subtitle}</div>" if subtitle else ""}
          </div>
        """,
        unsafe_allow_html=True,
    )
