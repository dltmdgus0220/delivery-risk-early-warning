from google_play_scraper import reviews, Sort
import pandas as pd
import time
from datetime import datetime
import argparse


COLUMNS = ['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'at']

def list_to_df(all_reviews: list) -> pd.DataFrame:
    df = pd.DataFrame(all_reviews)

    # userName 결측치 처리
    df.loc[df['userName'].isna(), 'userName'] = "Google 사용자"
    # 리뷰텍스트 결측치 처리
    df = df.dropna(subset=['content'])
    
    return df[COLUMNS]

def collect_reviews_by_num(app_id, num: int=1000, lang="ko", country="kr", batch_size=200, sleep_sec=0.2):
    all_reviews = []
    continuation_token = None
    seen = set()
    collected = 0

    while True:
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token
        )

        if not result:
            break

        stop = False

        for r in result:
            # 중복 제거 (수집 중 리뷰가 추가되어 중복된 리뷰가 들어갈 수도 있음)
            rid = r.get("reviewId")
            if rid not in seen:
                seen.add(rid)
                all_reviews.append(r)
                collected += 1
            
            if collected >= num:
                stop = True
                break

        if stop:
            break

        if continuation_token is None:
            break

        time.sleep(sleep_sec)

    return list_to_df(all_reviews)


def collect_reviews_by_date(app_id, start_date, end_date=None, lang="ko", country="kr", batch_size=200, sleep_sec=0.2):
    all_reviews = []
    continuation_token = None
    seen = set()

    start_date = start_date.replace(tzinfo=None)
    if end_date:
        end_date = end_date.replace(tzinfo=None)

    while True:
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token
        )

        if not result:
            break

        stop = False

        for r in result:
            review_dt = r["at"].replace(tzinfo=None) # timezone 제거

            # end_date가 있으면 end_date 이후 리뷰는 제외
            if end_date and review_dt > end_date:
                continue

            # start_date보다 오래된 리뷰가 나오면 중단 조건
            if review_dt < start_date:
                stop = True
                break
            
            # 중복 제거 (수집 중 리뷰가 추가되어 중복된 리뷰가 들어갈 수도 있음)
            rid = r.get("reviewId")
            if rid not in seen:
                seen.add(rid)
                all_reviews.append(r)

        if stop:
            break

        if continuation_token is None:
            break

        time.sleep(sleep_sec)

    return list_to_df(all_reviews)

