import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW # BERT에서 거의 표준으로 사용하는 옵티마이저

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# --- 0. 설정 ---

SEED = 42
MODEL_ID = "kykim/bert-kor-base"  # klue/bert-base, 참고:https://github.com/kiyoungkim1/LMkor
TRAIN_CSV_PATH = "data/out.csv"
CSV_PATH = "data/baemin_reviews_playstore_100000.csv"

TEXT_COL = "content"
LABEL_COL = "churn_intent_label"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5 # BERT에서 거의 표준으로 사용하는 학습률
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # gpu를 위한 코드

id2label = {0: "없음", 1: "약함", 2: "강함"}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # 경고 안뜨게 하기


# --- 1. 데이터로드 및 라벨처리 --- 

def drop_text(df: pd.DataFrame):
    df[TEXT_COL] = df[TEXT_COL].astype("string").str.strip()
    df[TEXT_COL] = df[TEXT_COL].replace(["", "NA", "NaN", "nan", "None", "<NA>"], np.nan)
    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    return df

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 텍스트 정리
    df = drop_text(df)

    # 문자열 라벨 -> 숫자 라벨
    df["label"] = np.select(
        [
            df[LABEL_COL] == "없음",
            df[LABEL_COL] == "약함",
            df[LABEL_COL] == "강함",
        ],
        [0, 1, 2],
        default=-1
    )

    # 라벨 변환 실패(-1) 제거
    df = df[df["label"] != -1].reset_index(drop=True)

    return df

