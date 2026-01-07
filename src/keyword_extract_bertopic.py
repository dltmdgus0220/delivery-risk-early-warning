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

