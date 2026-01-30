import pandas as pd


def risk_score_calc(df):
    num1 = len(df[df['churn_intent_label'] == 1])
    num2 = len(df[df['churn_intent_label'] == 2])
    return num2 * 1 + num1 * 0.9

