import os
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from torch.optim import AdamW # BERT에서 거의 표준으로 사용하는 옵티마이저
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast

from configs import MODEL_ID, MAX_LEN, DEVICE, EPS, id2label
from utils import set_seed, balanced_class_extract
from datasets import TrainTextDataset, InferTextDataset
from trainer import train_one_epoch, eval_model, predict_texts


# argparse
def build_argparser():
    p = argparse.ArgumentParser(description="이탈의도분류")
    p.add_argument("--input", required=True, help="입력 CSV 경로")
    p.add_argument("--save", required=True, help="모델 파라미터 저장 경로")
    p.add_argument("--text-col", default="content", help="텍스트 컬럼명")
    p.add_argument("--label-col", default="churn_intent_label", help="라벨 컬럼명")
    p.add_argument("--n", type=int, default=-1, help="클래스 밸런싱 (-1:미사용, 0:최솟값, n:n개)") # n개로 맞출 수 없을 시 자동으로 최솟값 사용
    p.add_argument("--model", type=int, default=1, help="사용할 모델 인덱스") # ["klue/bert-base", "kykim/bert-kor-base", "kykim/albert-kor-base", "kykim/funnel-kor-base", "electra-kor-base"]
    p.add_argument("--mode", required=True, choices=["train", "infer"], help="train/infer")
    p.add_argument("--seed", type=int, default=42, help="랜덤시드")
    p.add_argument("--batch", type=int, default=16, help="배치사이즈")
    p.add_argument("--epochs", type=int, default=5, help="에폭수")
    p.add_argument("--lr", type=float, default=2e-5, help="학습률")
    return p

