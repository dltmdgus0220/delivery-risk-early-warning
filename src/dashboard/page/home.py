import streamlit as st
import sqlite3
import asyncio
import ast
import altair as alt
import pandas as pd
from datetime import datetime, timedelta
from src.dashboard.pipeline import run_pipeline
from src.dashboard.util import parse_keywords


TABLE = 'data'
DATE_COL = 'at'

# --- 유틸 ---

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

def _count_between(conn, start_dt: datetime, end_dt: datetime) -> int:
    cur = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {TABLE}
        WHERE date({DATE_COL}) BETWEEN date(?) AND date(?)
        """,
        (start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")),
    )
    n = cur.fetchone()[0]
    cur.close()
    return int(n)

def _minmax_and_total(conn):
    cur = conn.execute(
        f"""
        SELECT MIN(date({DATE_COL})), MAX(date({DATE_COL})), COUNT(*)
        FROM {TABLE}
        """
    )
    mn, mx, total = cur.fetchone()
    cur.close()
    return mn, mx, int(total)

def _fmt_yy_mm_dd(s: str) -> str:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.strftime("%y.%m.%d")

def _fmt_k(n: int) -> str:
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return f"{n}"

