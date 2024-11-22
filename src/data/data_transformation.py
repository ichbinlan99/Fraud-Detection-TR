import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(".", "src")))

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features.feature_engineering import (
    generic_customer_spending_behaviour,
    general_customer_spending_bahaviour,
    get_merchant_risk_rolling_window,
)
from features.feature_transformation import encode, categorize_jobs


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# Load datasets
def load_data(path="./data/raw/tr_fincrime_test.csv"):
    """
    Load train and test datasets.
    """
    logger.info("Loading datasets...")
    test_df = pd.read_csv(path)
    logger.info(f"Loaded test and train datasets with shapes {test_df.shape}.")
    return test_df


# Utility functions
def load_encoder(path):
    """
    Load saved encoder from path.
    """
    logger.info(f"Loading encoder from {path}...")
    with open(path, "rb") as file:
        loaded_label_encoder = pickle.load(file)
    logger.info(f"Encoder loaded from {path}.")
    return loaded_label_encoder


def calculate_distance(row):
    """
    Calculate geodesic distance between customer and merchant locations.
    """
    cust_location = (row["lat"], row["long"])
    merch_location = (row["merch_lat"], row["merch_long"])
    return geodesic(cust_location, merch_location).miles


def assign_age_group(df, dob_col, trans_time_col, bins, labels):
    """
    Calculate age and assign age groups.
    """
    logger.info(f"Calculating age and assigning age groups...")
    df[dob_col] = pd.to_datetime(df[dob_col], errors="coerce")
    df[trans_time_col] = pd.to_datetime(df[trans_time_col], errors="coerce")
    df["age"] = (df[trans_time_col] - df[dob_col]).dt.days // 365
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels).astype("int64")
    logger.info("Age and age groups assigned.")
    return df


def categorize_and_encode_jobs(df, column, job_categories, path):
    """
    Categorize jobs and encode the categories dynamically.
    """
    logger.info("Categorizing and encoding jobs...")
    df = categorize_jobs(df, column, job_categories)
    with open(path, "rb") as file:
        loaded_label_encoder = pickle.load(file)
    df["job_encoded"] = loaded_label_encoder.transform(df[["job_category"]])
    logger.info("Jobs categorized and encoded.")
    return df


def scaleData(train_df, features):
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    with open("../saved_model/scaler/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return train_df


def compute_daily_fraud_stats(
    df, date_col="trans_date_trans_time", fraud_col="is_fraud", card_col="cc_num"
):
    """
    Computes daily fraud statistics from a transaction DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing transaction data.
    - date_col (str): Column name for the transaction datetime. Default is 'trans_date_trans_time'.
    - fraud_col (str): Column name indicating whether a transaction is fraudulent. Default is 'is_fraud'.
    - card_col (str): Column name for credit card numbers. Default is 'cc_num'.

    Returns:
    - pd.DataFrame: A DataFrame with daily transaction counts, fraud transaction counts,
      and unique fraudulent card counts.
    """
    # Ensure the datetime column is converted to datetime type
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract the date part
    df["trans_date"] = df[date_col].dt.date

    # Compute the number of transactions per day
    transactions_per_day = (
        df.groupby("trans_date").size().reset_index(name="num_transactions")
    )
    transactions_per_day["num_transactions_scaled"] = (
        transactions_per_day["num_transactions"] / 100
    )

    # Compute the number of fraudulent transactions and cards per day
    fraud_data = df[df[fraud_col] == 1]
    fraud_transactions_per_day = (
        fraud_data.groupby("trans_date")
        .size()
        .reset_index(name="num_fraud_transactions")
    )
    fraud_cards_per_day = (
        fraud_data.groupby("trans_date")[card_col]
        .nunique()
        .reset_index(name="num_fraud_cards")
    )

    # Merge the statistics into a single DataFrame
    daily_fraud_stats = (
        transactions_per_day.merge(
            fraud_transactions_per_day, on="trans_date", how="left"
        )
        .merge(fraud_cards_per_day, on="trans_date", how="left")
        .fillna(0)  # Fill NaN values with 0
    )

    return daily_fraud_stats
