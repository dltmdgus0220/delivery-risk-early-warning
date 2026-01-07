import pandas as pd
from bertopic import BERTopic
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# --- 1. 명사 추출 및 불용어 정의---

STOPWORDS = []
kiwi = Kiwi()

def extract_nouns(text):
    if not isinstance(text, str): 
        return ""
    result = kiwi.tokenize(text)
    # NNG(일반명사), NNP(고유명사), XR(어근) 추출
    return " ".join([t.form for t in result if t.tag in ['NNG', 'NNP', 'XR'] and len(t.form) > 1])


# --- 2. 데이터 로드 및 전처리 ---
df = pd.read_csv("data\out2.csv", encoding='utf-8-sig')
df['cleaned_text'] = df['content'].apply(extract_nouns)


# --- 3. 모델 설정 ---
# 한국어 임베딩에 최적화된 모델 사용
# Huggingface에 ko sentence transformer or ko sbert 검색
# jhgan/ko-sbert-multitask : torch v2.6 이상 필요, 제일 좋을 것 같은데 버전 안맞음
sentence_model = SentenceTransformer("kimseongsan/ko-sbert-384-reduced") # 문장 전체 임베딩 : 문장 내 단어들의 관계를 고려하여 각 문장마다 하나의 임베딩벡터 만들어줌.

# 키워드 추출 시 불용어 제거를 위한 Vectorizer 설정
vectorizer_model = CountVectorizer(stop_words=STOPWORDS, min_df=10, max_df=0.95) # min_df: 최소빈도, max_df: 최대빈도


# --- 4. BERTopic 모델 생성 및 학습 ---

# 임베딩(sentence_model) -> 차원축소(UMAP) -> 군집화(HDBSCAN) -> 키워드추출(c-TF-IDF)
topic_model = BERTopic(
    embedding_model=sentence_model,
    vectorizer_model=vectorizer_model,
    nr_topics=20, # auto : 토픽 개수 자동 조절
    verbose=True, # 진행바 활성
    low_memory=True, # 메모리 절약 모드
    # calculate_probabilities=False # 메모리 부족 방지를 위해 False 설정
)

print("토픽 모델링 학습 시작")
topics, probs = topic_model.fit_transform(df['cleaned_text'])

