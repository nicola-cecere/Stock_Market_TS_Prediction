import os
from typing import List, Tuple

import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def add_ticker_and_load_csv(file_path):
    file_path = "data/sp500/SP500.csv"
    folder_path = "data/sp500/csv/"
    output_file_path = "data/sp500/SP500.csv"

    ticker = os.path.basename(file_path).split(".")[0]
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df.insert(0, "Ticker", ticker)

    return df


def generete_unique_csv():
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


def split_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    return df


def encode_ticker(df: pd.DataFrame) -> pd.DataFrame:
    encoder = ce.BinaryEncoder(cols=["Ticker"])
    df_binary_encoded = encoder.fit_transform(df["Ticker"])
    df = df.join(df_binary_encoded)
    df.drop("Ticker", axis=1, inplace=True)
    return df


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    def categorize_month(month):
        if month in [4, 5, 7, 8, 9]:
            return "Bullish"
        elif month in [1, 10, 11, 12]:
            return "Bearish"
        else:
            return "Normal"

    df["Month_Category"] = df["Month"].apply(categorize_month)

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[["Month_Category"]]).toarray()

    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(["Month_Category"])
    )
    encoded_df.drop("Month_Category_Normal", axis=1, inplace=True)

    df_final = pd.concat([df, encoded_df], axis=1)
    df_final.drop(["Month_Category"], axis=1, inplace=True)
    return df_final
