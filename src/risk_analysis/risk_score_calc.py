import pandas as pd


def risk_score_calc(df):
    num1 = len(df[df['churn_intent_label'] == 1])
    num2 = len(df[df['churn_intent_label'] == 2])
    return round((num2 * 1 + num1 * 0.75) * (1/(len(df)+60)) , 2)


def monthly_risk_calc(df, date_col: str='at'):
    # 날짜 컬럼 처리
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce") # errors='coerce' : 날짜변환 실패시 결측값 저장
    df = df.dropna(subset=[date_col])

    # 월별 컬럼 생성
    df["month"] = df[date_col].dt.to_period("M").astype(str)

    # 월별로 묶어서 이탈지수 계산
    monthly_risk = (
        df.groupby("month")
        .apply(risk_score_calc, include_groups=False)
        .reset_index()
        .rename(columns={0: "risk_score"})
    )

    return monthly_risk


def main():
    df = pd.read_csv('data/out/baemin_reviews_playstore_99997_label_keyword.csv', encoding='utf-8-sig')
    monthly_risk = monthly_risk_calc(df)
    monthly_risk.to_csv('monthly_risk.csv', encoding='utf-8-sig')

if __name__ == "__main__":
    main()
