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

