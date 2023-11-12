import os
import warnings
from datetime import datetime
from typing import List, Tuple

import hdbscan  # Import HDBSCAN
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from scripts.utils import add_seasonality, encode_ticker, split_date

warnings.filterwarnings(action="ignore", category=FutureWarning, module="sklearn.*")


def objective(trial: optuna.Trial, X_train: pd.DataFrame) -> Tuple[float, float]:
    CPU_COUNT = os.cpu_count() - 4
    if CPU_COUNT is None:
        CPU_COUNT = 8

    # HDBSCAN parameters
    min_cluster_size = trial.suggest_int("min_cluster_size", 5, 50)
    min_samples = trial.suggest_int("min_samples", 5, 50)
    cluster_selection_epsilon = trial.suggest_float(
        "cluster_selection_epsilon", 0.0, 1.0
    )
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan"])

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        core_dist_n_jobs=CPU_COUNT,
    )

    clusters = model.fit_predict(X_train)

    print("Starting compute metrics")
    ch_score = calinski_harabasz_score(X_train, clusters)
    db_score = davies_bouldin_score(X_train, clusters)
    return db_score, ch_score


if __name__ == "__main__":
    load_dotenv()
    N_TRIALS = int(os.getenv("N_TRIALS", 100))

    df = pd.read_csv("data/sp500/SP500_training.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    # Feature engineering
    df = split_date(df)
    df = add_seasonality(df)
    df = encode_ticker(df)

    df.drop(columns=["Date"], inplace=True)

    X_train = df.copy()
    columns = X_train.select_dtypes(include=["float64"]).columns.tolist()
    means = X_train[columns].mean()
    stds = X_train[columns].std()

    # Normalize the data
    X_train[columns] = (X_train[columns] - means) / stds

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        study_name="hdbscan_sh_db_ch" + datetime.now().strftime("%Y%m%d%H%M%S"),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, X_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    print("Best trial:", study.best_trials)
