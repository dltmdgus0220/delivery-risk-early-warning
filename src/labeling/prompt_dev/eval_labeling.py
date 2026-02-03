import pandas as pd
import argparse
from sklearn.metrics import classification_report


# --- 0. 설정 ---
p = argparse.ArgumentParser(description="LLM 프롬프트 평가")
p.add_argument("--dir", required=True, help="폴더 경로")
p.add_argument("--save", action="store_true", help="틀린 데이터 저장 여부") # cli 입력 시 --save를 포함하면 true

args = p.parse_args()
    

# --- 1. 데이터로드 ---

df = pd.read_csv('data/prompt_dev/baemin_reviews_playstore_0_999_hardlabel.csv', encoding='utf-8-sig')
df1 = pd.read_csv(f'data/prompt_dev/{args.dir}/out_0_999_test1.csv', encoding='utf-8-sig')
df2 = pd.read_csv(f'data/prompt_dev/{args.dir}/out_0_999_test2.csv', encoding='utf-8-sig')
df3 = pd.read_csv(f'data/prompt_dev/{args.dir}/out_0_999_test3.csv', encoding='utf-8-sig')
df4 = pd.read_csv(f'data/prompt_dev/{args.dir}/out_0_999_test4.csv', encoding='utf-8-sig')
df5 = pd.read_csv(f'data/prompt_dev/{args.dir}/out_0_999_test5.csv', encoding='utf-8-sig')

print("[1 LLM 라벨링]")
print(df1['churn_intent_label'].value_counts())
print("\n[2 LLM 라벨링]")
print(df2['churn_intent_label'].value_counts())
print("\n[3 LLM 라벨링]")
print(df3['churn_intent_label'].value_counts())
print("\n[4 LLM 라벨링]")
print(df4['churn_intent_label'].value_counts())
print("\n[5 LLM 라벨링]")
print(df5['churn_intent_label'].value_counts())


# --- 2. 일관성 검증 ---

mask1 = (df1['churn_intent_label'] != df['label'])
mask2 = (df1['churn_intent'] != df2['churn_intent'])
mask3 = (df1['churn_intent'] != df3['churn_intent'])
mask4 = (df1['churn_intent'] != df4['churn_intent'])
mask5 = (df1['churn_intent'] != df5['churn_intent'])

print()
print("1과 정답 비교", mask1.sum())
print("1과 2 비교:", mask2.sum())
print("1과 3 비교:", mask3.sum())
print("1과 4 비교:", mask4.sum())
print("1과 5 비교:", mask5.sum())


# --- 3. 하드라벨과 비교 ---

print("\n[classification_report]")
print(classification_report(df['label'], df1['churn_intent_label']))

if args.save:
    df1['label'] = df['label']
    df_copy = df1[mask1][['content', 'score', 'churn_intent_label', 'churn_intent_reason', 'label']].copy()
    df_copy.to_csv(f'out_{args.dir}_comp.csv', encoding='utf-8-sig', escapechar='\\')