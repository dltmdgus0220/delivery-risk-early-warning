import pandas as pd
from keybert import KeyBERT
from kiwipiepy import Kiwi
from tqdm import tqdm


# --- 1. 모델 및 형태소 분석기 로드 ---

# 한국어를 포함한 다국어 지원 모델
kw_model = KeyBERT('distiluse-base-multilingual-cased-v1')
kiwi = Kiwi()

custom_stopwords = []

# --- 2. 전처리 ---

def extract_meaningful_words(text):
    if not isinstance(text, str):
        return ""
    
    # 분석에 방해되는 요소 제거 및 토큰화 
    result = kiwi.tokenize(text)
    
    # NNG(일반명사), NNP(고유명사), XR(어근), VA(형용사) 위주 추출
    # 너무 짧은 단어(1글자)는 제외하여 품질을 높입니다.
    words = [t.form for t in result if t.tag in ['NNG', 'NNP', 'XR', 'VA'] and len(t.form) > 1]
    
    return " ".join(words)

