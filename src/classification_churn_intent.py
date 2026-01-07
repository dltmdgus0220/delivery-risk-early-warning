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


# --- 2. 데이터셋 클래스 ---

class TrainTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.confidences = df['churn_intent_confidence'].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        confidence = self.confidences[idx]

        enc = self.tokenizer(
            text,
            truncation=True, # max_len보다 길면 뒷부분 삭제
            padding="max_length", # max_len으로 패딩해서 길이 맞추기
            max_length=self.max_len,
            return_tensors="pt", # pytorch tensor 형태로 결과 리턴
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0), # (1, MAX_LEN) -> (MAX_LEN,)
            "attention_mask": enc["attention_mask"].squeeze(0), # (1, MAX_LEN) -> (MAX_LEN,)
            "labels": torch.tensor(label, dtype=torch.long),
            "confidences": torch.tensor(confidence, dtype=torch.float)
        }
        return item


class InferTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df[TEXT_COL].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            truncation=True, # max_len보다 길면 뒷부분 삭제
            padding="max_length", # max_len으로 패딩해서 길이 맞추기
            max_length=self.max_len,
            return_tensors="pt", # pytorch tensor 형태로 결과 리턴
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0), # (1, MAX_LEN) -> (MAX_LEN,)
            "attention_mask": enc["attention_mask"].squeeze(0), # (1, MAX_LEN) -> (MAX_LEN,)
        }
        return item


# --- 3. 훈련/평가 ---

# 가중치 계산 함수
def weighted_cross_entropy(logits, labels, confidences):
    # reduction='none'으로 설정하여 각 샘플별 Loss를 먼저 구함, 원래는 배치별로 평균치를 구해서 계산함. 우리는 각 샘플별 confidence가 중요하기 때문에 none으로 설정
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits, labels)
    
    weighted_loss = (loss * confidences).mean()
    return weighted_loss

# train
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=True):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        confidences = batch['confidences'].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = weighted_cross_entropy(logits, labels, confidences)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader)) # 0으로 나누는 거 방지

# eval
@torch.no_grad() # 데코레이터
def eval_model(model, loader):
    model.eval()
    losses = []
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Eval", leave=True):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        confidences = batch['confidences'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        loss = weighted_cross_entropy(logits, labels, confidences)

        preds = torch.argmax(logits, dim=-1) # (batch_size, num_classes) -> (batch_size,)

        losses.append(loss.item())
        y_true.extend(batch["labels"].detach().cpu().numpy().tolist()) # numpy는 cpu에서만
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro") # 다중클래스일때는 average='macro'
    return {"loss": avg_loss, "acc": acc, "f1": f1}


# --- 4. 신규 데이터 예측 ---

@torch.no_grad()
def predict_texts(model, loader, threshold=0.6):
    model.eval()
    y_pred, y_conf, mask = [], [], []

    for batch in tqdm(loader, desc="Predict", leave=True):
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        out = model(**batch)

        logits = out.logits

        probs = torch.softmax(logits, dim=-1) # (batch_size, num_classes) -> (batch_size, num_classes)
        conf, preds = torch.max(probs, dim=-1) # (batch_size, num_classes) -> (batch_size,2), 가장 큰 확률과 인덱스를 반환

        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_conf.extend(conf.detach().cpu().numpy().tolist())

        # threshold 미만 row_id 저장
        low_mask = conf < threshold
        mask.extend(low_mask.detach().cpu().numpy().tolist())

    return y_pred, y_conf, mask


# --- 5. main ---

def main():
    set_seed(SEED)

    # 1) csv 로드
    new_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    new_df = drop_text(new_df)
    df = load_and_prepare(TRAIN_CSV_PATH)

    print("모델 :", MODEL_ID)
    print("전체 데이터 수 :", len(new_df))
    print("학습 데이터 수 :", len(df))
    print("이탈의도 클래스별 학습 데이터 수 :", df['label'].value_counts())

