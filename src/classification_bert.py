import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW # BERT에서 거의 표준으로 사용하는 옵티마이저

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# --- 0. 설정 ---

SEED = 42
MODEL_ID = "klue/bert-base"  # nlp4all/LMKor-BERT-base
CSV_PATH = "data/baemin_reviews_playstore_100000.csv"

TEXT_COL = "content" # 독립변수
LABEL_COL = "label" # 종속변수

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5 # BERT에서 거의 표준으로 사용하는 학습률
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # gpu를 위한 코드

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # 경고 안뜨게 하기

    main()