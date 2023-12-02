import sys

import pandas as pd
from dotenv import load_dotenv
from scripts.utils import add_seasonality

if __name__ == "__main__":
    load_dotenv()
    print(sys.path)

    df = pd.read_csv("data/sp500/csv/AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(by="Date")

    # drop usless columns
    df.drop(columns=["Adj Close"], inplace=True)

    # split date train, val, test 70, 20, 10
    df_train = df.iloc[: int(len(df) * 0.7)]
    df_val = df.iloc[int(len(df) * 0.7) : int(len(df) * 0.9)]
    df_test = df.iloc[int(len(df) * 0.9) :]

    # Feature engineering
    # df = add_seasonality(df)

    # standardize
