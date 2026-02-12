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

# 카드 css
def inject_card_css():
    st.markdown("""
    <style>
      .card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 14px 14px;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: transform .12s ease, box-shadow .12s ease;
      }
      .card:hover{
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
      }
      .kpi-label{
        font-size: 0.85rem;
        color: rgba(0,0,0,0.55);
        display:flex;
        align-items:center;
        gap:8px;
        margin-bottom: 4px;
      }
      .kpi-value{
        font-size: 2.1rem;
        font-weight: 750;
        letter-spacing: -0.02em;
        line-height: 1.1;
      }
      .kpi-sub{
        margin-top: 6px;
        font-size: 0.82rem;
        color: rgba(0,0,0,0.45);
      }
      .class-title{
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 6px;
      }
      .class-count{
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
      }
      .badge{
        display:inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 650;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(0,0,0,0.03);
      }
      .row{
        display:flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
      }
      .leftbar{
        border-left: 6px solid var(--barcolor);
        padding-left: 12px;
      }
    </style>
    """, unsafe_allow_html=True)

# 리뷰수/이탈지수 카드
def kpi_card(label: str, value: str, icon: str = "📌", sub: str | None = None):
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-label">{icon}<span>{label}</span></div>
          <div class="kpi-value">{value}</div>
          {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# 클래스별 비율 카드
def class_card(class_name: str, count: int, ratio: float, bar_color: str = "#3B82F6", delta_pp: float | None = None):
    st.markdown(
        f"""
        <div class="card leftbar" style="--barcolor:{bar_color};">
          <div class="class-title">{class_name}</div>
          <div class="class-count">{count:,}건</div>
          <div class="row">
            <span class="badge">{ratio:.2f}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 월별 추이 꺾은선 그래프 시각화
def plot_monthly_line(df_m: pd.DataFrame, y_col: str, y_title: str, tick_count: int = 8):
    """
    df_m 컬럼: month(YYYY-MM), count, risk_score
    """
    base = alt.Chart(df_m).encode(
        x=alt.X(
            "month:N",
            title="월",
            axis=alt.Axis(labelAngle=0, tickCount=tick_count)  # ✅ xtick 개수 제어
        )
    )

    line = base.mark_line().encode(
        y=alt.Y(f"{y_col}:Q", title=y_title),
    )

    points = base.mark_point(filled=True, size=40).encode(
        y=alt.Y(f"{y_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip("month:N", title="월"),
            alt.Tooltip(f"{y_col}:Q", title=y_title),
        ],
    )

    # ✅ hover 시 세로 룰 + 값 표시 (인터랙티브 감성)
    hover = alt.selection_point(fields=["month"], nearest=True, on="mouseover", empty=False)

    rule = base.mark_rule(opacity=0.2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("month:N", title="월"),
            alt.Tooltip(f"{y_col}:Q", title=y_title),
        ],
    ).add_params(hover)

    chart = (line + points + rule).properties(height=260).interactive()  # ✅ 줌/팬 유지
    return chart


# --- 2. 사이드바 ---

def render_sidebar(today):
    st.sidebar.subheader("🔄 데이터 관리")

    db_path = st.sidebar.text_input(
        "DB 경로",
        value="demo.db"
    )
    st.session_state['db_path'] = db_path

     # 데이터 갱신 버튼
    if st.sidebar.button("데이터 갱신", use_container_width=True):
        status = st.sidebar.empty()
        status.info("파이프라인 실행 중...")

        try:
            conn = sqlite3.connect(db_path)
            flag = asyncio.run(run_pipeline(conn, today))

            # 파이프라인 후 min/max 다시 조회
            mn_new, mx_new, _ = _minmax_and_total(conn)
            conn.close()

            if mn_new and mx_new:
                mn_new, mx_new = _to_date(mn_new), _to_date(mx_new)

                # 기간 자동 갱신
                st.session_state["start_dt"] = mn_new
                st.session_state["end_dt"] = mx_new

            status.empty()

            if flag == 0:
                st.sidebar.success("데이터 갱신 완료!")
            else:
                st.sidebar.success("이미 최신 데이터입니다.")

            st.rerun()

        except Exception as e:
            status.empty()
            st.sidebar.error(f"실행 실패: {e}")

    # 기간 선택
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 오버뷰 필터")
    try:
        conn = sqlite3.connect(db_path)
        mn, mx, total = _minmax_and_total(conn)
    finally:
        conn.close()

    mn, mx = _to_date(mn), _to_date(mx)

    st.session_state.setdefault("start_dt", mn)
    st.session_state.setdefault("end_dt", mx)

    start_dt = st.sidebar.date_input("기간 시작", value=st.session_state["start_dt"], min_value=mn, max_value=mx)
    end_dt = st.sidebar.date_input("기간 종료", value=st.session_state["end_dt"], min_value=mn, max_value=mx)

    if st.sidebar.button("적용"):
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        st.session_state["start_dt"], st.session_state["end_dt"] = start_dt, end_dt
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("DB 요약")

    try:
        conn = sqlite3.connect(db_path)

        cur_s, cur_e = _month_range(today, 0)      # 이번 달
        prev_s, prev_e = _month_range(today, -1)  # 지난 달

        cur_cnt = len(fetch_period_df(db_path, DATA_TABLE, cur_s, cur_e))
        prev_cnt = len(fetch_period_df(db_path, DATA_TABLE, prev_s, prev_e))

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
        "db_path": db_path,
        "start_dt": st.session_state["start_dt"],
        "end_dt": st.session_state["end_dt"],
    }


# --- 3. 메인 ---

def render(cfg: dict, today: datetime):
    set_korean_font()
    inject_card_css()

    db_path = cfg["db_path"]
    start_dt = cfg['start_dt']
    end_dt = cfg['end_dt']

    # 데이터로드
    df_data = fetch_period_df(db_path, DATA_TABLE, start_dt, end_dt)
    df_data['keywords'] = df_data['keywords'].map(parse_keywords)

    # 월별 집계를 위한 컬럼 추가
    df_data['month'] = df_data['at'].map(lambda x: x[:7])

    # 이탈지수계산
    risk_score = risk_score_calc(df_data)

    # 클래스분리
    df_confirmed = df_data[df_data['churn_intent_label'] == 2].copy()
    df_complaint = df_data[df_data['churn_intent_label'] == 1].copy()
    df_positive = df_data[df_data['churn_intent_label'] == 0].copy()

    # 대시보드 렌더링
    st.markdown("## 🛵 '배달의민족' 이탈 리스크 분석 대시보드")
    st.markdown("### Overview")
    st.caption(f"분석 기간: {start_dt:%Y-%m-%d} ~ {end_dt:%Y-%m-%d}")

    st.divider()

    # 1행 (집계요약, 추이 시각화)
    left, right = st.columns([1, 1.8], gap="medium")

    # 집계 요약
    with left:
        st.markdown("#### 📌 수집 현황")

        c1, c2 = st.columns(2)

        with c1:
            kpi_card("리뷰수", f"{len(df_data):,}건", icon="🗂️", sub=f"최근 적재 날짜: {date.today():%Y-%m-%d}")
        with c2:
            kpi_card("이탈지수", f"{risk_score:.2f}", icon="⚠️", sub="0에 가까울수록 안정")
        
        st.divider()

        st.markdown("##### 클래스별 분포")
        r1, r2, r3 = st.columns(3)

        with r1:
            ratio_confirmed = round((len(df_confirmed) / len(df_data)) * 100, 2)
            class_card("'확정'", len(df_confirmed), ratio_confirmed, bar_color="#EF4444")
        with r2:
            ratio_complaint = round((len(df_complaint) / len(df_data)) * 100, 2)
            class_card("'불만'", len(df_complaint), ratio_complaint, bar_color="#F59E0B")
        with r3:
            ratio_positive = round((len(df_positive) / len(df_data)) * 100, 2)
            class_card("'없음'", len(df_positive), ratio_positive, bar_color="#10B981")

    # 추이 시각화
    with right:
        st.markdown("#### 📈 월별 추이")

        rows = []
        for m, g in df_data.groupby("month", sort=True):
            rows.append({
                "month": m,
                "count": int(len(g)),
                "risk_score": float(risk_score_calc(g)) if len(g) else 0.0,
            })

        df_m = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

        if df_m.empty:
            st.info("선택 기간에 표시할 데이터가 없어요.")
        else:
            metric = st.selectbox("지표 선택", ["리뷰수", "이탈지수"], index=0,)

            # 간단 요약(최근월 기준)
            latest = df_m.iloc[-1]
            if metric == "이탈지수":
                st.caption(f"최근월({latest['month']}) 이탈지수: {latest['risk_score']:.2f}")
                chart = plot_monthly_line(df_m, "count", "리뷰수(건)")
            else:
                st.caption(f"최근월({latest['month']}) 리뷰수: {int(latest['count']):,}건")
                chart = plot_monthly_line(df_m, "risk_score", "이탈지수")
            st.altair_chart(chart, use_container_width=True)
    
    # 클래스별 키워드 TopN
    st.divider()
    st.subheader("클래스별 키워드 TopN")

    top_n = st.slider("Top N", 5, 30, 10, 1)

    # 카운터
    counter_confirmed = keyword_count(df_confirmed)
    counter_complaint = keyword_count(df_complaint)
    counter_positive = keyword_count(df_positive)

    # topn 키워드
    topn_list = {
        "확정": top_n_keywords_extract(counter_confirmed, n=top_n),
        "불만": top_n_keywords_extract(counter_complaint, n=top_n),
        "없음": top_n_keywords_extract(counter_positive, n=top_n),
    }

    buckets = ["확정", "불만", "없음"]
    cols = st.columns(3, gap="large")

    for i, label in enumerate(buckets):
        with cols[i]:
            st.markdown(f"#### '{label}' 키워드 Top{top_n}")

            topn = topn_list[label]

            if not topn:
                st.caption("데이터가 없습니다.")
            else:
                df_kw = pd.DataFrame(topn, columns=["keyword", "cnt"])

                bar = (
                    alt.Chart(df_kw)
                    .mark_bar()
                    .encode(
                        y=alt.Y(
                            "keyword:N",
                            sort="-x",
                            axis=alt.Axis(title=None)
                        ),
                        x=alt.X(
                            "cnt:Q",
                            axis=alt.Axis(title="빈도")
                        ),
                        tooltip=[
                            alt.Tooltip("keyword:N", title="키워드"),
                            alt.Tooltip("cnt:Q", title="빈도"),
                        ],
                    )
                )
                st.altair_chart(bar, use_container_width=True)