import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, max_len: int):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].astype(int).tolist()
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

        return {
            "input_ids": enc["input_ids"].squeeze(0), # (1, MAX_LEN) -> (MAX_LEN,)
            "attention_mask": enc["attention_mask"].squeeze(0), # (1, MAX_LEN) -> (MAX_LEN,)
            "labels": torch.tensor(label, dtype=torch.long)
        }

