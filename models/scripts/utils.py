import os
import warnings
from typing import List, Tuple

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from category_encoders import BinaryEncoder

# from fracdiff.sklearn import FracdiffStat
from numpy.fft import irfft, rfft
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers.legacy import Adam


def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 1):
        end = i + n_steps
        seq_x, seq_y = data.iloc[i:end, :].values, data.iloc[end]["Close"]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def add_ticker_and_load_csv(file_path):
    file_path = "data/sp500/SP500.csv"
    folder_path = "data/sp500/csv/"
    output_file_path = "data/sp500/SP500.csv"

    ticker = os.path.basename(file_path).split(".")[0]
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df.insert(0, "Ticker", ticker)

    return df


def load_csv(ticker):
    folder_path = "data/sp500/csv/"
    file_path = f"{folder_path}{ticker}.csv"

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    return df


def trigonometric_date_encoding(df: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    """Encode date as sin and cos of the day of the week from a date object.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str, optional): The column name with the date to encode. Defaults to "f_1".

    Returns:
        pd.DataFrame: The dataframe with the encoded date.
            The new columns are called sin_date and cos_date.
            The original column is not dropped.
    """
    # Convert the column to datetime
    df[column] = pd.to_datetime(df[column], format="%d-%m-%Y")

    # Extract the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = df[column].dt.dayofweek

    # Calculate sin and cos
    date_sin = np.sin(day_of_week * (2.0 * np.pi / 7.0))
    date_cos = np.cos(day_of_week * (2.0 * np.pi / 7.0))

    # Create a DataFrame with the new columns
    encoded_dates = pd.DataFrame({"sin_date": date_sin, "cos_date": date_cos})

    # Concatenate the new columns with the original dataframe
    result_df = pd.concat([df, encoded_dates], axis=1)

    return result_df


def create_lags(df, n_lags):
    def fill_with_first_close(lag_df, n_lags):
        for lag in range(1, n_lags + 1):
            first_valid_index = lag_df["Close"].first_valid_index()
            first_valid_value = (
                lag_df.loc[first_valid_index, "Close"]
                if first_valid_index is not None
                else 0
            )
            lag_df[f"lag_{lag}"] = lag_df[f"lag_{lag}"].fillna(first_valid_value)
        return lag_df

    lag_df = df.copy()

    # Now proceed with sorting and creating lag features
    lag_df.sort_values(by=["Date"], inplace=True)

    for lag in range(1, n_lags + 1):
        lag_df[f"lag_{lag}"] = lag_df["Close"].shift(lag)

    return fill_with_first_close(lag_df, n_lags)


def generete_unique_csv(folder_path, output_file_path):
    csv_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".csv")
    ]
    combined_df = pd.concat(
        (add_ticker_and_load_csv(file) for file in csv_files), ignore_index=True
    )
    sorted_df = combined_df.sort_values(["Ticker", "Date"])

    sorted_df.to_csv(output_file_path, index=False)


def create_missing_values_csv(df):
    # Extract the year from the 'Date' column
    df["Year"] = df["Date"].dt.year

    # Count the total number of rows for each ticker
    total_counts = df.groupby("Ticker").size()

    # Count the missing values for 'Adjusted Close' for each ticker
    missing_counts = df[df["Adjusted Close"].isnull()].groupby("Ticker").size()

    # Calculate the overall percentage of missing values for each ticker
    overall_missing_percentage = (missing_counts / total_counts * 100).reset_index(
        name="Overall Missing Percentage"
    )

    # Calculate the number of missing values for each ticker, each year
    missing_counts_yearly = (
        df[df["Adjusted Close"].isnull()].groupby(["Ticker", "Year"]).size()
    )

    # Calculate the total number of rows for each ticker, each year
    total_counts_yearly = df.groupby(["Ticker", "Year"]).size()

    # Calculate the percentage of missing values for each ticker, each year
    missing_percentage_yearly = (
        missing_counts_yearly / total_counts_yearly * 100
    ).reset_index(name="Missing Percentage")

    # Merge the overall missing percentage with the yearly statistics
    ticker_missing_stats = pd.merge(
        overall_missing_percentage, missing_percentage_yearly, on="Ticker", how="right"
    )

    # Display the statistics for each ticker
    ticker_missing_stats = ticker_missing_stats[
        ["Ticker", "Overall Missing Percentage", "Year", "Missing Percentage"]
    ]
    ticker_missing_stats["Total Counts"] = total_counts_yearly.values
    ticker_missing_stats.to_csv("data/sp500/missing_values.csv")


def create_number_rows_by_year(df):
    # Count the number of rows for each ticker per year
    ticker_year_distribution = df.groupby(["Ticker", "Year"]).size().unstack().fillna(0)

    ticker_year_distribution.to_csv("data/sp500/numberrows.csv")


def remove_outliers_in_batches(
    df: pd.DataFrame, columns: List[str], coefficient: int
) -> pd.DataFrame:
    # Copy of df
    new_df = df.copy()

    # Add columns to the new dataframe to flag outliers
    for col in columns:
        new_df[col + "_Outlier"] = False

    # Process each ticker separately
    for ticker in new_df["Ticker"].unique():
        ticker_data = new_df[new_df["Ticker"] == ticker]
        ticker_data = ticker_data.sort_values(by="Date")

        # Get the range of years
        start_year = ticker_data["Date"].dt.year.min()
        end_year = ticker_data["Date"].dt.year.max()

        # Process in batches of up to 10 years
        for start in range(start_year, end_year, 10):
            end = min(start + 10, end_year + 1)
            batch = ticker_data[
                (ticker_data["Date"].dt.year >= start)
                & (ticker_data["Date"].dt.year < end)
            ]

            # Compute the mean and std dev for the batch
            stats = batch[columns].agg(["mean", "std"])

            # Find and flag outliers in the batch
            for col in columns:
                mean = stats[col]["mean"]
                std = stats[col]["std"]
                outlier_condition = abs(batch[col] - mean) > (coefficient * std)
                batch_indices = batch[outlier_condition].index
                new_df.loc[batch_indices, col + "_Outlier"] = True

    return new_df


def encode_ticker(df: pd.DataFrame) -> pd.DataFrame:
    encoder = ce.BinaryEncoder(cols=["Ticker"])
    df_binary_encoded = encoder.fit_transform(df["Ticker"])
    df_binary_encoded = df_binary_encoded.astype("Int8")
    df = df.join(df_binary_encoded)
    df.drop("Ticker", axis=1, inplace=True)
    return df


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    def categorize_month(month):
        if month in [8, 9, 11]:
            return "Bullish"
        elif month in [1, 2, 3, 4, 5]:
            return "Bearish"
        else:
            return "Normal"

    df["Month_Category"] = df["Month"].apply(categorize_month)

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[["Month_Category"]]).toarray()

    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(["Month_Category"])
    )

    df_final = pd.concat([df, encoded_df], axis=1)
    df_final.drop(["Month_Category"], axis=1, inplace=True)
    return df_final


def frac_diff_stationarity(train, test):
    # Make a copy of the train data inside the function to avoid modifying the original dataframe
    train_internal_copy = train.copy()

    fd = FracdiffStat()
    fd.fit(train_internal_copy[["Close"]].values)

    # Replace the 'Close' column with the transformed data in the copy
    train_internal_copy["Close"] = fd.transform(train_internal_copy[["Close"]].values)
    test["Close"] = fd.transform(test[["Close"]].values)

    # Return the modified copy and test
    return train_internal_copy, test


def split_data_frame(df, train_frac=0.7, val_frac=0.2):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df


def apply_moving_average_for_roc(dataframe, ma_type="ema", ma_window=50, roc_window=20):
    def bin_roc_adjusted(roc_value):
        if pd.isna(roc_value):
            return "Unknown"  # Handling NaN values separately
        elif roc_value > 10:
            return "Very High Positive"
        elif roc_value > 5:
            return "High Positive"
        elif roc_value > 1:
            return "Low Positive"
        elif roc_value > -1:
            return "Neutral"
        elif roc_value > -5:
            return "Low Negative"
        else:
            return "High Negative"

    df = dataframe.copy()

    if ma_type == "ma":
        # Calculate the 50-Day Moving Average
        df["50-Day MA"] = df["Close"].rolling(window=ma_window).mean()
        # Calculate the Rate of Change for the 50-Day Moving Average
        df["Rate of Change"] = df["50-Day MA"].pct_change(periods=roc_window) * 100
    elif ma_type == "ema":
        # Calculate the 50-Day Exponential Moving Average
        df["50-Day MA"] = df["Close"].ewm(span=ma_window, adjust=False).mean()
        # Calculate the Rate of Change for the 50-Day Exponential Moving Average
        df["Rate of Change"] = df["50-Day MA"].pct_change(periods=roc_window) * 100
    else:
        raise ValueError(
            "Invalid ma_type. Choose 'ma' for Moving Average or 'ema' for Exponential Moving Average"
        )

    # Bin the Rate of Change
    df["ROC"] = df["Rate of Change"].apply(bin_roc_adjusted)

    # Use BinaryEncoder to encode the 'ROC' column
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        from category_encoders import BinaryEncoder

        encoder = BinaryEncoder(cols=["ROC"], drop_invariant=True)
        df_encoded = encoder.fit_transform(df[["ROC"]])

    # Concatenate the encoded 'ROC' column with the original dataframe
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=["ROC", "Rate of Change", "50-Day MA"], inplace=True)

    # Return the dataframe with the encoded 'ROC' column added

    return df
