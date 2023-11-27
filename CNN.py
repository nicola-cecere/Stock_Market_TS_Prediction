import os
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from scripts.utils import add_seasonality, DataWindow, split_data_frame, split_date, compile_and_fit

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense

import warnings
warnings.filterwarnings('ignore')

KERNEL_WIDTH = 3
LABEL_WIDTH = 21
INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1


if __name__ == "__main__":
    load_dotenv()
    N_TRIALS = int(os.getenv("N_TRIALS", 100))

    df = pd.read_csv("data/sp500/csv/AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(by="Date")

    train_df, val_df, test_df = split_data_frame(df)

# Feature engineering
    train_df = split_date(train_df)
    # train_df = add_seasonality(train_df)
    train_df.drop(columns=["Date"], inplace=True)

    val_df = split_date(val_df)
    # val_df = add_seasonality(val_df)
    val_df.drop(columns=["Date"], inplace=True)

    test_df = split_date(test_df)
    # test_df = add_seasonality(test_df)
    test_df.drop(columns=["Date"], inplace=True)

# Feature Scaling
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
    val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
    test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

    cnn_multi_window = DataWindow(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, train_df=train_df, val_df=val_df, test_df=test_df, shift=21, label_columns=['Close'])

    cnn_model = Sequential([
        Conv1D(32, activation='relu', kernel_size=(KERNEL_WIDTH)),
        Dense(units=32, activation='relu'),
        Dense(1, kernel_initializer=tf.initializers.zeros),
    ])

    history = compile_and_fit(cnn_model, cnn_multi_window)

    val_performance = {}
    performance = {}

    val_performance['CNN'] = cnn_model.evaluate(cnn_multi_window.val)
    performance['CNN'] = cnn_model.evaluate(cnn_multi_window.test, verbose=0)