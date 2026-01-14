import pandas as pd
from keybert import KeyBERT
from kiwipiepy import Kiwi
from tqdm import tqdm


# --- 1. 모델 및 형태소 분석기 로드 ---

# 한국어를 포함한 다국어 지원 모델
kw_model = KeyBERT('distiluse-base-multilingual-cased-v1')
kiwi = Kiwi()

custom_stopwords = []

