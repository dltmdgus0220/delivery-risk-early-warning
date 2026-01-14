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


# --- 3. 키워드 도출 ---

def get_refined_keywords_safe(df, label_filter="강함", top_n=10, diversity=0.7, sample_size=5000):
    """
    메모리 에러 방지를 위해 샘플링 방식을 도입한 함수입니다.
    """
    # 라벨 필터링
    target_df = df[df['churn_intent_label'] == label_filter].copy()
    
    if len(target_df) == 0:
        print(f"라벨 {label_filter}에 해당하는 데이터가 없습니다.")
        return []

    # 샘플링
    if len(target_df) > sample_size:
        print(f"{sample_size}건 무작위 샘플링")
        target_df = target_df.sample(n=sample_size, random_state=42)

    print(f"분석 시작 (데이터 수: {len(target_df)}건)...")

    # 전처리
    tqdm.pandas()
    target_df['cleaned_text'] = target_df['content'].progress_apply(extract_meaningful_words)
    
    # 텍스트 병합
    all_text = " ".join(target_df['cleaned_text'].tolist())

    # KeyBERT 실행
    try:
        keywords = kw_model.extract_keywords(
            all_text,
            keyphrase_ngram_range=(1, 2),
            stop_words=custom_stopwords,
            top_n=top_n,
            use_mmr=True,
            diversity=diversity
        )
    except Exception as e:
        print(f"에러 발생: {e}. diversity 값을 낮추거나 sample_size를 더 줄여보세요.")
        return []
    
    return keywords

