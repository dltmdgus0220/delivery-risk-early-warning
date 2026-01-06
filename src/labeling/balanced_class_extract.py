import pandas as pd

df = pd.read_csv('data/baemin_reviews_playstore_100000.csv', encoding='utf-8-sig')
df_pos = df[df['sentiment_label'] == 1].head(200).copy()
df_neg = df[df['sentiment_label'] == -1].head(300).copy()
df_merge = pd.concat((df_pos, df_neg))
df_merge.to_csv('data/balanced_class_reviews_500.csv', encoding='utf-8-sig', escapechar='\\')