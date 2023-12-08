import os
from typing import List, Tuple

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from fracdiff.sklearn import FracdiffStat
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers.legacy import Adam


def add_ticker_and_load_csv(file_path):
    file_path = "data/sp500/SP500.csv"
    folder_path = "data/sp500/csv/"
    output_file_path = "data/sp500/SP500.csv"

    ticker = os.path.basename(file_path).split(".")[0]
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df.insert(0, "Ticker", ticker)

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


def split_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month.astype("Int8")
    df["Day"] = df["Date"].dt.day.astype("Int8")
    return df


def encode_ticker(df: pd.DataFrame) -> pd.DataFrame:
    encoder = ce.BinaryEncoder(cols=["Ticker"])
    df_binary_encoded = encoder.fit_transform(df["Ticker"])
    df_binary_encoded = df_binary_encoded.astype("Int8")
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


import pandas as pd
from category_encoders import BinaryEncoder


def apply_ma_for_roc(dataframe, ma_window=50, roc_window=20):
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

    # Calculate the 50-Day Moving Average
    df["50-Day MA"] = df["Close"].rolling(window=ma_window).mean()

    # Calculate the Rate of Change for the 50-Day Moving Average
    df["MA Rate of Change"] = df["50-Day MA"].pct_change(periods=roc_window) * 100

    # Bin the Rate of Change
    df["ROC"] = df["MA Rate of Change"].apply(bin_roc_adjusted)

    # Use BinaryEncoder to encode the 'ROC' column
    encoder = BinaryEncoder(cols=["ROC"], drop_invariant=True)
    df_encoded = encoder.fit_transform(df[["ROC"]])

    # Concatenate the encoded 'ROC' column with the original dataframe
    df = pd.concat([df, df_encoded], axis=1)
    df.drop("ROC", axis=1, inplace=True)
    df.drop("MA Rate of Change", axis=1, inplace=True)
    df.drop("50-Day MA", axis=1, inplace=True)

    # Return the dataframe with the encoded 'ROC' column added
    return df


class DataWindow:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=None,
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Name of the column that we wish to predict
        self.label_columns = label_columns
        if label_columns is not None:
            # Create a dictionary with the name and index of the label column. This will be used for plotting.
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        # Create a dictionary with the name and index of each column. This will be used to separate the features from the target variable
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # The slice function returns a slice object that specifies how to slice a sequence.
        # In this case, it says that the input slice starts at 0 and ends when we reach the input_width.
        self.input_slice = slice(0, input_width)
        # Assign indices to the inputs. These are useful for plotting.
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # Get the index at which the label starts. In this case, it is the total window size minus the width of the label.
        self.label_start = self.total_window_size - self.label_width
        # The same steps that were applied for the inputs are applied for labels.
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        # Slice the window to get the inputs using the input_slice defined in __init__.
        inputs = features[:, self.input_slice, :]
        # Slice the window to get the labels using the labels_slice defined in __init__
        labels = features[:, self.labels_slice, :]

        # If we have more than one target, we stack the labels.
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )
        # The shape will be [batch, time, features].
        # At this point,we only specify the time dimension and allow the batch and feature dimensions to be defined later.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="Close", max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        # Plot the inputs. They will  appear as a continuous blue line with dots.
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f"{plot_col} [scaled]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            # Plot the labels or actual. They will appear as green squares.
            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                marker="s",
                label="Labels",
                c="green",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                # Plot the predictions. They will appear as red crosses.
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="red",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Date (Day)")
        plt.ylabel("Closing price (USD)")

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            # Pass in the data. This corresponds to our training set, validation set, or test set.
            data=data,
            # Targets are set to None, as they are handled by the split_to_input_labels function.
            targets=None,
            # Define the total length of s the array, which is equal to the total window length.
            sequence_length=self.total_window_size,
            # Define the number of timesteps separating each sequence. In our case, we want the sequences to be consecutive, so sequence_stride=1.
            sequence_stride=1,
            # Shuffle the sequences. Keep in mind that the data is still in chronological order. We are simply shuffling the order of the sequences, which makes the model more robus
            shuffle=True,
            # Define the number of sequences in a single batch
            batch_size=32,
        )
        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        # Get a sample batch of data for plotting purposes. If the sample batch does not exist, we’ll retrieve a sample batch and cache it
        result = getattr(self, "_sample_batch", None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result


# The function takes a model, and a window of data from the DataWindow class.
# The patience: is the number of epochs after which the model should stop training if the validation loss does not improve;
# max_epochs: sets a maximum number of epochs to train the model.
def compile_and_fit(model, window, patience=3, max_epochs=50):
    # Early stopping occurs if 3 consecutive epochs do not decrease the validation loss, as set by the patience parameter
    # The validation loss is tracked to determine if we should apply early stopping or not.
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

    # The MSE is used as the loss function.
    model.compile(
        loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()]
    )  # the MAE as an evaluation metric to compare the performance of our models

    # The model is fit on the training set.
    history = model.fit(
        window.train,
        epochs=max_epochs,  # The model can train for at most 50 epochs, as set by the max_epochs parameter.
        validation_data=window.val,
        callbacks=[early_stopping],
    )  # early_stopping is passed as a callback. If the validation loss does not decrease after 3 consecutive epochs, the model stops training. This avoids overfitting.
    return history
