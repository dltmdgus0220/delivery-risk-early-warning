import pandas as pd
import argparse
import os

def main():
    p = argparse.ArgumentParser(description="")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--pos", type=int, default=400, help="라벨링할 긍정 클래스 샘플 수")
    p.add_argument("--neg", type=int, default=600, help="라벨링할 부정 클래스 샘플 수")
    p.add_argument("--out", required=True, help="저장 CSV 경로")
    p.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    p.add_argument("--shuffle", action="store_true", help="샘플링 시 셔플 여부") # 인자를 호출하면 True, 안하면 False

    args = p.parse_args()
    
    df = pd.read_csv(args.csv, encoding='utf-8-sig')

    pos_df = df[df["sentiment_label"] == 1]
    neg_df = df[df["sentiment_label"] == -1]

    pos_n = min(args.pos, len(pos_df))
    neg_n = min(args.neg, len(neg_df))

    if args.shuffle:
        pos_df = pos_df.sample(n=pos_n, random_state=args.seed).copy()
        neg_df = neg_df.sample(n=neg_n, random_state=args.seed).copy()
    else:
        pos_df = pos_df.head(pos_n).copy()
        neg_df = neg_df.head(neg_n).copy()

    df_merge = pd.concat((pos_df, neg_df))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_merge.to_csv(args.out, encoding='utf-8-sig', index=False, escapechar='\\')
    print(f"총 {len(df_merge)}개 데이터 저장 완료")

if __name__ == "__main__":
    main()
