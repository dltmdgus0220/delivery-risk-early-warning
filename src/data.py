from google_play_scraper import reviews, Sort
import time
import pandas as pd

# --- 0. 상수선언 ---

PACKAGE_LIST = ['com.sampleapp','com.coupang.mobile.eats','com.fineapp.yogiyo','com.shinhan.o2o'] # 배민, 쿠팡이츠, 요기요, 땡겨요
PACKAGE_NAME_KOR = ['배달의민족', '쿠팡이츠', '요기요', '땡겨요']
PACKAGE_NAME_ENG = ['baemin', 'coupangeats', 'yogiyo', 'ddangyo']
PACKAGE_NUM = 3
NUM_DATA = 7000

APP_ID = PACKAGE_LIST[PACKAGE_NUM]
APP_NAME_KOR = PACKAGE_NAME_KOR[PACKAGE_NUM]
APP_NAME_ENG = PACKAGE_NAME_ENG[PACKAGE_NUM]


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


# --- 3. csv 파일 저장 ---

columns = ['app', 'platform', 'reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'at']
df[columns].to_csv(f'data/{APP_NAME_ENG}_reviews_playstore_{NUM_DATA}.csv', index=False, encoding="utf-8-sig") # utf-8-sig:윈도우+엑셀에서 한글 깨짐 방지
