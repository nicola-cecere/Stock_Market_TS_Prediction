import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# from scripts.utils import add_seasonality
from sklearn.preprocessing import StandardScaler


def create_sequences(data, t):
    X_train, y_train = [], []
    i = 0
    while i + 2 * t <= len(data):  # Ensure enough data for both X_train and y_train
        X_train.append(data.iloc[i : i + t, :].values)
        y_train.append(data.iloc[i + t : i + t + t]["Close"].values)
        i += t
    return np.array(X_train), np.array(y_train)


if __name__ == "__main__":
    load_dotenv()

    df = pd.read_csv("data/sp500/csv/AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(by="Date")

    # Extract the year, month, and day as separate columns
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date", "Adjusted Close"], inplace=True)

    # df = add_seasonality(df)

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size, :]
    test = df.iloc[train_size:, :]

    scaler = StandardScaler()
    scaler.fit(train)
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)

    # Create sequences of t timesteps with d dimensions
    t = 21  # 6 months
    X_train, y_train = create_sequences(train, t)
    X_test, y_test = create_sequences(test, t)

    # X_train is shaped as [samples, time steps, features]
    # y_train is shaped as [samples, t labels]

    model = Sequential()
    model.add(
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]))
    )  # Adjusted for dynamic input shape
    model.add(Dense(t))  # Output layer with 't' units, one for each label

    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (Mean Squared Error): {test_loss}")

    # Plotting the actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_sample, label="Actual", color="blue", marker="o")
    plt.plot(
        y_pred_sample, label="Predicted", color="red", linestyle="dashed", marker="x"
    )
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
