import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW # BERT에서 거의 표준으로 사용하는 옵티마이저

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# --- 0. 설정 ---

SEED = 42
MODEL_ID = "kykim/bert-kor-base"  # klue/bert-base, 참고:https://github.com/kiyoungkim1/LMkor
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


# --- 1. 데이터로드 및 라벨처리 --- 

def load_and_prepare(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['label'] = np.select([df['sentiment_label'] == 1, df['sentiment_label'] == -1], [1, 0], default=-1) # 결측치 및 중립은 -1
    df = df[df['label'].isin([1,0])].copy()
    return df


# --- 2. 데이터셋 클래스 ---

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df[LABEL_COL].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

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
        }
        return item


# --- 3. 훈련/평가 ---

# train
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=True):
        batch = {k: v.to(DEVICE) for k, v in batch.items()} # DEVICE 위에 데이터 올리기

        optimizer.zero_grad()
        out = model(**batch) # **batch: 딕셔너리 언패킹
        loss = out.loss

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
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)

        loss = out.loss
        logits = out.logits

        preds = torch.argmax(logits, dim=-1) # (batch_size, num_classes) -> (batch_size,)

        losses.append(loss.item())
        y_true.extend(batch["labels"].detach().cpu().numpy().tolist()) # numpy는 cpu에서만
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


# --- 4. 신규 데이터 eval ---

@torch.no_grad()
def predict_texts(model, tokenizer, texts, max_len=128):
    model.eval()

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    pos_prob = probs[:, 1].detach().cpu().numpy() # positive 확률
    return pred, pos_prob


# --- 5. 기타 함수 ---

def loss_visualization(history: List[dict]):
    history_df = pd.DataFrame(history)

    epochs = history_df['epoch']
    train_loss = history_df['train_loss']
    val_loss = history_df['val_loss']
    val_acc = history_df['val_acc']
    val_f1 = history_df['val_f1']

    # loss 곡선
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")

    # epoch별 annotation (val_acc, val_f1)
    for e, vl, acc, f1 in zip(epochs, val_loss, val_acc, val_f1):
        plt.annotate(
            f"acc={acc:.3f}\nf1={f1:.3f}",
            xy=(e, vl),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss with Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 불확실샘플 수 확인
@torch.no_grad()
def count_uncertain_by_entropy(model, loader, df, threshold=0.65):
    model.eval()
    rows = []
    start = 0
    uncertain_count = 0
    total = 0
    
    for batch in loader:
        batch_size = batch['labels'].size(0)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)

        probs = torch.softmax(out.logits, dim=-1)
        pos_probs = probs[:, 1].detach().cpu().numpy()
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).detach().cpu().numpy()

        mask = (entropy >= threshold)
        uncertain_count += mask.sum().item()
        total += probs.size(0)

        selected = df.iloc[start:start + batch_size][mask].copy()
        selected["pos_prob"] = pos_probs[mask]
        selected["entropy"] = entropy[mask]
        rows.append(selected)

        start += batch_size

    example_df = (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())

    return uncertain_count, total, example_df


# --- 6. main ---

def main():
    set_seed(SEED)

    # 1) csv 로드
    df = load_and_prepare(CSV_PATH)
    # NA , "", " ", None, NaN 등 처리하기
    df["content"] = df["content"].astype("string").str.strip()
    df["content"] = df["content"].replace(["", "NA", "NaN", "nan", "None", "<NA>"], np.nan)
    df = df.dropna(subset=["content"]).reset_index(drop=True)

    # 2) train/val/test 분할 (8/1/1)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df[LABEL_COL]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df[LABEL_COL]
    )
    print("모델 :", MODEL_ID)
    print("전체 데이터 수 :", len(df))
    print("긍/부정 데이터 수 :", df['label'].value_counts())
    print(f"train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # 3) tokenizer/model 생성
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True) # use_fast=True: Rust 기반 fast tokenizer 사용, 옛날모델은 미지원.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2, use_safetensors=True, force_download=True)
    model.to(DEVICE)

    # dataloader
    train_loader = DataLoader(TextDataset(train_df, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df, tokenizer, MAX_LEN),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TextDataset(test_df, tokenizer, MAX_LEN),
                             batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)

    # 4) train + validation
    history = []
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer)
        val_metrics = eval_model(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_f1': val_metrics['f1']
        })

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}")
        
    # test 평가
    test_metrics = eval_model(model, test_loader)
    print(f"[TEST] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f}")

    # loss 시각화
    if EPOCHS > 1:
        loss_visualization(history)

    # 불확실샘플 수 확인
    total_loader = DataLoader(TextDataset(df, tokenizer, MAX_LEN),
                             batch_size=BATCH_SIZE, shuffle=False)
    uncertain_cnt, total_cnt, df_example = count_uncertain_by_entropy(model, total_loader, df)
    print(f"[Entropy] 불확실 샘플 수: {uncertain_cnt}/{total_cnt} "
        f"({uncertain_cnt/total_cnt:.2%})")
    df_example.to_csv('/data/baemin_reviews_playstore_complex_sentiment.csv', encoding='utf-8-sig', escapechar="\\")

    # 5) 신규 데이터 eval
    new_texts = [
        "배달이 너무 늦고 고객센터도 답이 없어요",
        "배달도 빠르고 음식도 따뜻하게 와서 만족합니다",
        "배차 시스템이 문제인지 배달이 너무 느려요. 그래도 혜택은 만족합니다.",
        "배달은 늦었는데 쿠폰은 좋음",
        "상담은 별로지만 기사님은 친절",
    ]
    pred, pos_prob = predict_texts(model, tokenizer, new_texts, max_len=MAX_LEN)
    for t, p, pr in zip(new_texts, pred, pos_prob):
        label = "positive" if p == 1 else "negative"
        print(f"\nTEXT: {t}\nPRED: {label} (pos_prob={pr:.3f})")


if __name__ == "__main__":
    main()