import pandas as pd

df1 = pd.read_csv('data/v1/out_0_999_test1.csv', encoding='utf-8-sig')
df2 = pd.read_csv('data/v1/out_0_999_test2.csv', encoding='utf-8-sig')
df3 = pd.read_csv('data/v1/out_0_999_test3.csv', encoding='utf-8-sig')
df4 = pd.read_csv('data/v1/out_0_999_test4.csv', encoding='utf-8-sig')
df5 = pd.read_csv('data/v1/out_0_999_test5.csv', encoding='utf-8-sig')

print("[1 LLM 라벨링]")
print(df1['churn_intent'].value_counts())
print("\n[2 LLM 라벨링]")
print(df2['churn_intent'].value_counts())
print("\n[3 LLM 라벨링]")
print(df3['churn_intent'].value_counts())
print("\n[4 LLM 라벨링]")
print(df4['churn_intent'].value_counts())
print("\n[5 LLM 라벨링]")
print(df5['churn_intent'].value_counts())

mask1 = (df1['churn_intent'] != df2['churn_intent'])
mask2 = (df1['churn_intent'] != df3['churn_intent'])
mask3 = (df1['churn_intent'] != df4['churn_intent'])
mask4 = (df1['churn_intent'] != df5['churn_intent'])

print()
print("1과 2 비교:", mask1.sum())
print("1과 3 비교:", mask2.sum())
print("1과 4 비교:", mask3.sum())
print("1과 5 비교:", mask4.sum())
