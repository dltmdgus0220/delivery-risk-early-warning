from google_play_scraper import reviews, Sort
import time
import pandas as pd
import numpy as np

# --- 0. 상수선언 ---

APP_ID = 'com.sampleapp'
APP_NAME_KOR = '배달의민족'
APP_NAME_ENG = 'baemin'
NUM_DATA = 100000


# --- 1. 데이터 수집 ---

all_reviews = []
token = None
seen = set()

while True: # 한번의 호출가능한 수가 한정되어있으므로 반복해야함. count=10,000이라고해서 10,000개 가져오는거 아님.
    batch, token = reviews(
        APP_ID,
        lang="ko", # 언어
        country="kr", # 국가
        count=200,
        sort=Sort.NEWEST,
        continuation_token=token
    )
    # 중복 제거 (수집 중 리뷰가 추가되어 중복된 리뷰가 들어갈 수 있음)
    add = 0 # 추가 여부 확인
    for r in batch:
        rid = r.get("reviewId")
        if rid not in seen:
            seen.add(rid)
            all_reviews.append(r)
            add += 1
    if add == 0:
        break
    
    if len(all_reviews) >= NUM_DATA:
        all_reviews = all_reviews[:NUM_DATA]
        break

    time.sleep(0.5) # 너무 빠르게 호출하면 에러 발생할 수 있음.

print("수집한 리뷰 수:", len(all_reviews))


# --- 2. 데이터프레임 변환 ---

df = pd.DataFrame(all_reviews)
df['app'] = APP_NAME_KOR
df['platform'] = 'playstore'

# 라벨링
conditions = [
    df['score'] >= 4,
    df['score'] == 3,
    df['score'] <= 2
]

df['sentiment_label'] = np.select(conditions, [1, 0, -1], default=99)
df['sentiment'] = np.select(conditions, ['positive', 'neutral', 'negative'], default='unknown')


# --- 3. csv 파일 저장 ---

columns = ['app', 'platform', 'reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'at', 'sentiment_label', 'sentiment']
df[columns].to_csv(
    f'data/{APP_NAME_ENG}_reviews_playstore_{NUM_DATA}.csv',
    index=False,
    encoding="utf-8-sig", # utf-8-sig: 윈도우+엑셀에서 한글 깨짐 방지
    escapechar="\\" # 이스케이프 문자 지정
    ) 
