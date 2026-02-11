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

def render_cooccur_card(target_kw, cls, co_list, base_n):
    subtitle = f"{cls} · '{target_kw}' · 포함 리뷰 {base_n:,}건 기준"
    card_container("🤝 동시발생 키워드", subtitle)

    if not co_list:
        st.markdown(
            "<div style='padding:12px;color:#64748b;'>동시발생 키워드가 없습니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    for r in co_list:
        st.markdown(
            f"""
            <div class="kw-row">
              <div class="kw-left" title="{target_kw} + {r['keyword']}">
                {target_kw} + {r['keyword']}
              </div>
              <div class="kw-right">
                {int(r['count'])}건 | {r['ratio']:.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# 동시발생 키워드 섹션 렌더링
def render_cooccur_panel(df_cur: pd.DataFrame, co_cls: str, co_target_kw: str):
    # 선택 안 했을 때: 카드 형태로 안내도 통일
    if not co_target_kw or co_target_kw == "(선택)":
        card_container("🤝 동시발생 키워드", "사이드바에서 기준 키워드를 선택하세요.")
        st.markdown(
            "<div style='padding:12px;color:#64748b;'>표시할 결과가 없습니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    df_cls = filter_df_by_class(df_cur, co_cls)
    co_list, base_n = cooccur_top(df_cls, target_kw=co_target_kw, top_k=10)

    render_cooccur_card(
        target_kw=co_target_kw,
        cls=co_cls,
        co_list=co_list,
        base_n=base_n,
    )

# 리뷰 드릴다운
def render_drilldown_panel(df_cur: pd.DataFrame, dd_cls: str, dd_target_kw: str, limit: int = 50):
    # 카드 헤더 통일
    subtitle = f"{dd_cls} · 키워드: {dd_target_kw if dd_target_kw and dd_target_kw != '(선택)' else '미선택'}"
    card_container("🔍 드릴다운", subtitle)
    st.markdown("")

    if not dd_target_kw or dd_target_kw == "(선택)":
        st.markdown(
            "<div style='padding:12px;color:#64748b;'>사이드바에서 키워드를 선택하면 리뷰 리스트를 표시합니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    df_cls = filter_df_by_class(df_cur, dd_cls)

    if df_cls.empty:
        st.markdown(
            "<div style='padding:12px;color:#64748b;'>해당 클래스 데이터가 없습니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    # keywords는 list[str]이라고 가정
    mask = df_cls["keywords"].apply(lambda ks: dd_target_kw in ks)
    df_hit = df_cls[mask].copy()

    if df_hit.empty:
        st.markdown(
            "<div style='padding:12px;color:#64748b;'>선택한 키워드가 포함된 리뷰가 없습니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    # 시간/텍스트 컬럼
    time_col = "at"
    text_col = "content"

    # at 정렬 + 표시용 포맷
    if time_col:
        df_hit[time_col] = pd.to_datetime(df_hit[time_col], errors="coerce")
        df_hit = df_hit.sort_values(time_col, ascending=False)
        df_hit["작성시간"] = df_hit[time_col].dt.strftime("%Y-%m-%d %H:%M")
    else:
        df_hit["작성시간"] = ""

    out = df_hit[["작성시간", text_col]].rename(columns={text_col: "리뷰"}).head(limit)

    st.dataframe(out, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)  # card_container 닫기

# --- 4행 ---
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
def _extract_text_list(summary_dict: dict, key: str) -> list[str]:
    """
    summary_dict[key]가
    - [{"text": "...", "importance": n}, ...] 형태면 text만 추출
    - ["...","..."] 형태면 그대로 문자열만
    """
    if not summary_dict:
        return []
    items = summary_dict.get(key, [])
    out = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                t = str(it.get("text", "")).strip()
                if t:
                    out.append(t)
            else:
                t = str(it).strip()
                if t:
                    out.append(t)
    return out

def render_summary_section(title: str, obj):
    data = _as_dict(obj)
    st.markdown(f"##### {title}")

    if not data:
        st.caption("요약 데이터가 없습니다.")
        return

    # 섹션별 텍스트만 뽑아서 출력
    sections = [
        ("문제 상황", "situations"),
        ("기존 대응에 대한 평가", "evaluations"),
        ("소비자들이 원하는 대응", "solutions"),
    ]

    for head, k in sections:
        texts = _extract_text_list(data, k)
        with st.container(border=True):
            st.markdown(f"**{head}**")
            if not texts:
                st.caption("내용이 없습니다.")
            else:
                for t in texts:
                    st.write(f"- {t}")


# --- 2. 사이드바 ---

# 기본 사이드바 렌더링
def render_sidebar(today: datetime):
    with st.sidebar:    
        st.markdown("### 📅 월 선택")

        # 기준: 지난달
        y, m = today.year, today.month - 1
        if m == 0:
            y -= 1
            m = 12

        # 최근 24개월 생성 (기준달부터)
        months = []

        for _ in range(24):
            months.append(f"{y:04d}-{m:02d}")
            m -= 1
            if m == 0:
                y -= 1
                m = 12

        selected_month = st.selectbox(
            "분석 기준 월",
            options=months,
            index=0, # 항상 지난달이 첫 번째
        )

        st.markdown("---")
        st.markdown("### 🔑 키워드 TopN 설정")

        topn_class = st.radio(
            "대상 클래스",
            options=["확정", "불만", "확정+불만"],
            horizontal=True,
            key="topn_target",
        )

        topn_n = st.slider(
            "TopN (N)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="topn_n",
        )

    return {
        "yyyymm": selected_month,
        "year": int(selected_month.split("-")[0]),
        "month": int(selected_month.split("-")[1]),
        "topn_class": topn_class,
        "topn_n": topn_n,
    }

# 동시발생 키워드 사이드바 렌더링
def render_cooccur_sidebar(df_cur: pd.DataFrame):
    with st.sidebar:
        st.markdown("### 🤝 동시발생 키워드 설정")

        co_cls = st.radio(
            "대상 클래스",
            ["확정", "불만", "확정+불만"],
            horizontal=True,
            key="co_cls",
        )

        df_cls = filter_df_by_class(df_cur, co_cls)
        suggest_list, _ = top_keywords_for_suggest(df_cls, top_k=20)

        co_target_kw = st.selectbox(
            "기준 키워드 (Top20 추천)",
            options=suggest_list,
            index=0,
            key="co_target_kw",
        )

    return {
        "co_cls": co_cls,
        "co_target_kw": co_target_kw,
    }

# 드릴다운 사이드바 렌더링
def render_drilldown_sidebar(df_cur: pd.DataFrame):
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 드릴다운 설정")

        dd_cls = st.radio(
            "대상 클래스",
            ["확정", "불만", "확정+불만"],
            horizontal=True,
            key="dd_cls",
        )

        df_cls = filter_df_by_class(df_cur, dd_cls)
        suggest_list, _ = top_keywords_for_suggest(df_cls, top_k=20)

        dd_target_kw = st.selectbox(
            "키워드 검색 (Top20 추천)",
            options=suggest_list,
            index=0,
            key="dd_target_kw",
        )

        dd_limit = st.slider(
            "표시 개수",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="dd_limit",
        )

    return {
        "dd_cls": dd_cls,
        "dd_target_kw": dd_target_kw,
        "dd_limit": dd_limit,
    }


# --- 3. 메인 ---

def render(cfg_base: dict, today):
    set_korean_font()
    inject_css()
    inject_keyword_list_css()
    db_path = st.session_state.get("db_path")

    # 기준 월
    cur_dt = datetime.strptime(cfg_base["yyyymm"], "%Y-%m")
    prev_dt = cur_dt - relativedelta(months=1)

    cur_yyyymm = cur_dt.strftime("%Y-%m")
    prev_yyyymm = prev_dt.strftime("%Y-%m")

    # 데이터로드
    df_cur = fetch_month_df(db_path, "data", cur_yyyymm)
    df_cur["keywords"] = df_cur["keywords"].apply(parse_keywords)
    df_prev = fetch_month_df(db_path, "data", prev_yyyymm)
    df_prev["keywords"] = df_prev["keywords"].apply(parse_keywords)
    df_cur_summary = fetch_month_df(db_path, "summary", cur_yyyymm)
    df_prev_summary = fetch_month_df(db_path, "summary", prev_yyyymm)

    # 동시발생/드릴다운 사이드바 추가 및 통합
    cfg_co = render_cooccur_sidebar(df_cur)
    cfg_dd = render_drilldown_sidebar(df_cur)
    cfg = {**cfg_base, **cfg_co, **cfg_dd}

    # 데이터분리
    df_cur_confirmed = df_cur[df_cur['churn_intent_label'] == 2].copy()
    df_cur_complaint = df_cur[df_cur['churn_intent_label'] == 1].copy()
    df_cur_positive = df_cur[df_cur['churn_intent_label'] == 0].copy()
    df_prev_confirmed = df_prev[df_prev['churn_intent_label'] == 2].copy()
    df_prev_complaint = df_prev[df_prev['churn_intent_label'] == 1].copy()
    df_prev_positive = df_prev[df_prev['churn_intent_label'] == 0].copy()

    # 클래스 비율 계산
    ratio_cur_confirmed = round(len(df_cur_confirmed)/len(df_cur)*100, 1)
    ratio_cur_complaint = round(len(df_cur_complaint)/len(df_cur)*100, 1)
    ratio_cur_positive = round(len(df_cur_positive)/len(df_cur)*100, 1)
    ratio_prev_confirmed = round(len(df_prev_confirmed)/len(df_prev)*100, 1)
    ratio_prev_complaint = round(len(df_prev_complaint)/len(df_prev)*100, 1)
    ratio_prev_positive = round(len(df_prev_positive)/len(df_prev)*100, 1)

    st.caption(
        f"※ 모든 증감 수치는 지난달({prev_dt.year % 100:02d}년 {prev_dt.month:02d}월) 대비 기준입니다."
    )

    year, month = cfg["year"], cfg["month"]

    st.markdown("## 🔑 키워드 중심 분석")
    st.markdown(f"### {year % 100:02d}년 {month:02d}월 데이터 분석")

    st.divider()

    # 1행 (데이터수/이탈지수/클래스별분포)
    st.markdown("#### 📌 수집 현황")

    delta_cnt = len(df_cur) - len(df_prev)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card(
            label="리뷰 수",
            value=f"{len(df_cur):,}건",
            delta_text=f"{delta_cnt:+,}건",
            delta_is_good=(delta_cnt >= 0),
        )

    with c2:
        churn_value = df_cur_summary.iloc[0]['risk_score']
        churn_delta = churn_value - df_prev_summary.iloc[0]['risk_score']
        kpi_card(
            label="이탈지수",
            value=f"{churn_value:.2f}",
            delta_text=f"{churn_delta:+.2f}",
            delta_is_good=(churn_delta < 0),
        )

    with c3:
        delta_p = round(ratio_cur_confirmed - ratio_prev_confirmed, 1)
        class_mini_card("'확정'", len(df_cur_confirmed), ratio_cur_confirmed, delta_p, (delta_p < 0))
    with c4:
        delta_p = round(ratio_cur_complaint - ratio_prev_complaint, 1)
        class_mini_card("불만", len(df_cur_complaint), ratio_cur_complaint, delta_p, (delta_p < 0))
    with c5:
        delta_p = round(ratio_cur_positive - ratio_prev_positive, 1)
        class_mini_card("없음", len(df_cur_positive), ratio_cur_positive, delta_p, (delta_p > 0))

    st.markdown("---")
    # 2행 (left: 키워드 TopN, right: 키워드 추이)
    left, right = st.columns([1, 1.4], gap="small")

    with left:
        if cfg["topn_class"] == '확정':
            topn, selected_kw = render_top_keywords_bar_plotly(
                df=df_cur_confirmed,
                title="'확정' 키워드 TopN",
                top_n=cfg["topn_n"],
            )
        elif cfg["topn_class"] == '불만':
            topn, selected_kw = render_top_keywords_bar_plotly(
                df=df_cur_complaint,
                title="'불만' 키워드 TopN",
                top_n=cfg["topn_n"],
            )

        else: # 확정+불만
            topn, selected_kw = render_top_keywords_bar_plotly(
                df=pd.concat([df_cur_confirmed, df_cur_complaint], ignore_index=True),
                title="'확정+불만'키워드 TopN",
                top_n=cfg["topn_n"],
            )

    with right:
        # selected_kw가 없으면 안내
        if not selected_kw:
            st.info("왼쪽 TopN 막대에서 키워드를 선택하면 추이를 표시합니다.", icon="👈")
        else:
            min_yyyymm, max_yyyymm = get_min_max_yyyymm(db_path)
            months_11 = build_11mo_window(cur_yyyymm, min_yyyymm=min_yyyymm, max_yyyymm=max_yyyymm)

            trend_df = monthly_keyword_ratio(
                db_path=db_path,
                months=months_11,
                keyword=selected_kw,
                cls=cfg["topn_class"],
            )

            render_keyword_trend_line(
                df_trend=trend_df,
                title=f"'{selected_kw}' 키워드 비중 추이 ({cfg['topn_class']})",
                center_yyyymm=cur_yyyymm,
            )

    st.markdown("---")
    # 3행 (left: top - 신규 키워드, bottom - 급증 키워드, mid: 동시발생 키워드, right: 드릴다운)
    # ✅ 신규/급증 계산 (3행 직전이나 3행 안에서 한번만)
    df_cur_cls = filter_df_by_class(df_cur, cfg["topn_class"])
    df_prev_cls = filter_df_by_class(df_prev, cfg["topn_class"])

    counter_cur = keyword_count(df_cur_cls)
    counter_prev = keyword_count(df_prev_cls)

    new_list, surged_list = detect_keyword_changes(
        counter_prev=counter_prev,
        counter_cur=counter_cur,
        threshold=0.03,     # 예: 3%p 이상 증가를 급증으로 (너 데이터에 맞게 조절)
        min_cur_count=5,
    )

    left, mid, right = st.columns([1, 1, 2], gap="small")

    with left:
        render_keyword_list_card("🆕 신규 키워드", new_list, top_k=5, mode="new")
        st.markdown("") # 간격
        render_keyword_list_card("📈 급증 키워드", surged_list, top_k=5, mode="surge")

    with mid:
        render_cooccur_panel(
            df_cur=df_cur,
            co_cls=cfg["co_cls"],
            co_target_kw=cfg["co_target_kw"],
        )
    with right:
        render_drilldown_panel(
            df_cur=df_cur,
            dd_cls=cfg["dd_cls"],
            dd_target_kw=cfg["dd_target_kw"],
            limit=cfg["dd_limit"],
        )

    st.markdown("---")
    # 4행 (요약, 드릴다운)
    bottom_left, bottom_right = st.columns([1.2, 1.5], gap="large")

    # 요약
    with bottom_left:
        st.markdown(f"#### 🧠 '{topn[0][0]}' 중심 요약")

        view_mode = st.radio(
            "표시할 요약 선택",
            options=["확정", "불만"],
            horizontal=True,
            label_visibility="collapsed",
            key="summary_view_mode",
        )

        if df_cur_summary is None or df_cur_summary.empty:
            st.info("요약 데이터가 없습니다.")
        else:
            row0 = df_cur_summary.iloc[0]

            confirmed_obj = row0.get("summary_confirmed", None)
            complaint_obj = row0.get("summary_complaint", None)

            if view_mode == "확정":
                render_summary_section("'확정' 리뷰 분석", confirmed_obj)
            elif view_mode == "불만":
                render_summary_section("'불만' 리뷰 분석", complaint_obj)

    # 드릴다운
    with bottom_right:
        st.markdown(f"#### 🔍 '{topn[0][0]}' 드릴다운")

        if df_cur_summary is None or df_cur_summary.empty:
            st.info("요약 데이터가 없어 근거 리뷰를 찾을 수 없습니다.", icon="🧩")
        else:
            row0 = df_cur_summary.iloc[0]

            # summary 객체에서 reason_id 모으기
            reason_ids = []

            if view_mode in ["확정"]:
                conf_obj = _as_dict(row0.get("summary_confirmed", None))
                reason_ids += _as_id_list(conf_obj.get("reason_id", None))

            elif view_mode in ["불만"]:
                comp_obj = _as_dict(row0.get("summary_complaint", None))
                reason_ids += _as_id_list(comp_obj.get("reason_id", None))

            # 중복 제거(순서 유지)
            seen = set()
            reason_ids = [x for x in reason_ids if not (str(x) in seen or seen.add(str(x)))]

            if not reason_ids:
                st.info("선택된 요약에 근거 리뷰 ID(reason_id)가 없습니다.", icon="🧩")
            else:
                # df_cur에서 id/날짜/라벨/텍스트 컬럼 자동 탐색
                id_col = "reviewId"
                at_col = "at"
                label_col = "churn_intent"
                text_col = "content"

            if id_col is None:
                st.error("df_cur에서 리뷰 id 컬럼을 찾지 못했습니다. (예: id/review_id)")
            else:
                # 타입 맞추기: reason_ids가 문자열일 수도 있어서 문자열 비교로 통일
                df_tmp = df_cur.copy()
                df_tmp["_id_str"] = df_tmp[id_col].astype(str)
                id_set = set(str(x) for x in reason_ids)

                df_drill = df_tmp[df_tmp["_id_str"].isin(id_set)].copy()

                if df_drill.empty:
                    st.warning("reason_id로 매칭되는 리뷰를 df_cur에서 찾지 못했습니다.")
                else:
                    # 보기용 컬럼 구성
                    out = pd.DataFrame()
                    out["날짜"] = df_drill[at_col].astype(str) if at_col else ""
                    out["클래스"] = df_drill[label_col].astype(str) if label_col else ""
                    out["리뷰"] = df_drill[text_col].astype(str) if text_col else ""

                    # 날짜 컬럼이 있으면 정렬
                    if at_col:
                        try:
                            df_drill["_at_dt"] = pd.to_datetime(df_drill[at_col])
                            out = out.loc[df_drill.sort_values("_at_dt", ascending=False).index]
                        except Exception:
                            pass

                    st.caption(f"근거 리뷰 {len(out)}건 (reason_id 기준)")
                    st.dataframe(out, use_container_width=True, hide_index=True, height=520)

    
