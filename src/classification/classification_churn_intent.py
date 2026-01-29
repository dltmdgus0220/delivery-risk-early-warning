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

# train
def train_pipeline(args):
    set_seed(args.seed)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    model_id = MODEL_ID[args.model]

    print("모델 :", model_id)
    print("전체 데이터 수 :", len(df))
    print("이탈의도 클래스별 분포 :", df[args.label_col].value_counts())

    # split
    train_df, tmp = train_test_split(
        df, test_size=0.2, random_state=args.seed, shuffle=True, stratify=df[args.label_col]
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.5, random_state=args.seed, shuffle=True, stratify=tmp[args.label_col]
    )
    print(f"train/val/test : {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # balancing (train only)
    if args.n == 0:
        num = min(train_df[args.label_col].value_counts())
        train_df = balanced_class_extract(train_df, args.label_col, num, args.seed)
        print("[train set 클래스 밸런싱 완료]")
    elif args.n > 0:
        num = min(train_df[args.label_col].value_counts())
        train_df = balanced_class_extract(train_df, args.label_col, min(num, args.n), args.seed)
        print("[train set 클래스 밸런싱 완료]")

    if args.n >= 0:
        print("train set 데이터 수 :", len(train_df))
        print("이탈의도 클래스별 분포 :", train_df[args.label_col].value_counts())

    # tokenizer/model
    if args.model == 2: # albert
        tokenizer = BertTokenizerFast.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True) # use_fast=True: Rust 기반 fast tokenizer 사용, 옛날모델은 미지원.
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, use_safetensors=True).to(DEVICE) # safetensors: 가중치를 저장하는 포맷

    # dataloader
    train_loader = DataLoader(
        TrainTextDataset(train_df, tokenizer, args.text_col, args.label_col, MAX_LEN),
        batch_size=args.batch, shuffle=True
    )
    val_loader = DataLoader(
        TrainTextDataset(val_df, tokenizer, args.text_col, args.label_col, MAX_LEN),
        batch_size=args.batch, shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best = {"recall": -1.0, "precision": -1.0, "loss": float("inf")}

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = eval_model(model, val_loader, DEVICE)

        cp = val_metrics["class2_precision"]
        cr = val_metrics["class2_recall"]
        vl = val_metrics["loss"]

        print(
            f"[Epoch {epoch}] train_loss={tr_loss:.4f} | "
            f"val_loss={vl:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_class2_precision={cp:.4f} val_class2_recall={cr:.4f}"
        )

        # save best model (class2 recall)
        improved = (
            (cr > best["recall"] + EPS) or
            (abs(cr - best["recall"]) <= EPS and cp > best["precision"] + EPS) or
            (abs(cr - best["recall"]) <= EPS and abs(cp - best["precision"]) <= EPS and vl < best["loss"] - EPS)
        )

        if improved:
            best.update({"recall": cr, "precision": cp, "loss": vl})

            os.makedirs(args.save, exist_ok=True)
            model.save_pretrained(args.save)
            tokenizer.save_pretrained(args.save)

            print(f"Saved best model | recall={cr:.4f}, precision={cp:.4f}, val_loss={vl:.4f}")

    # test with best checkpoint
    best_model = AutoModelForSequenceClassification.from_pretrained(args.save).to(DEVICE)
    best_tokenizer = AutoTokenizer.from_pretrained(args.save, use_fast=True)

    test_loader = DataLoader(
        InferTextDataset(test_df, best_tokenizer, args.text_col, MAX_LEN),
        batch_size=args.batch, shuffle=False
    )

    y_true = test_df[args.label_col]
    y_pred = predict_texts(best_model, test_loader, DEVICE)

    print("\n[혼동 행렬]")
    print(confusion_matrix(y_true, y_pred))
    print("\n[분류 리포트]")
    print(classification_report(y_true, y_pred, target_names=["없음", "불만", "확정"]))

# inference
def infer_pipeline(args):
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    tokenizer = AutoTokenizer.from_pretrained(args.save, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.save).to(DEVICE)

    loader = DataLoader(InferTextDataset(df, tokenizer, args.text_col, MAX_LEN), batch_size=args.batch, shuffle=False)
    preds = predict_texts(model, loader, DEVICE)
    df['churn_intent'] = [id2label[p] for p in preds]
    df['churn_intent_label'] = preds

    df.to_csv('out.csv', encoding='utf-8-sig', escapechar='\\')


def main():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    args = build_argparser().parse_args()

    if args.mode == "train":
        train_pipeline(args)
    else:
        infer_pipeline(args)


if __name__ == "__main__":
    main()