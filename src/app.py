import os
import sys
import streamlit as st
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dashboard.page import home, overview, summary, analysis, operation

st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)

TODAY = datetime.now(timezone.utc)
PAGES = {
    "Home": home,
    "Overview": overview,
    "Summary": summary,
    "Analysis": analysis,
    "Operation": operation,
}

def main():
    st.sidebar.title("📊 Dashboard")

    # 상단: 페이지 선택
    page_name = st.sidebar.radio("페이지", list(PAGES.keys()))
    page = PAGES[page_name]

    st.sidebar.divider()

    # 하단: 페이지별 메뉴 (있으면)
    cfg = {}
    if hasattr(page, "render_sidebar"):
        cfg = page.render_sidebar(TODAY) or {}

    st.sidebar.divider()

    # 본문 렌더
    if hasattr(page, "render"):
        page.render(cfg, TODAY)
    else:
        st.error("페이지에 render(cfg) 함수가 필요합니다.")

if __name__ == "__main__":
    main()
