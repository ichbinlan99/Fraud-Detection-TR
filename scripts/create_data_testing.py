import json
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(".", "src")))

import pandas as pd

from data.data_transformation import (
    assign_age_group,
    calculate_distance,
    categorize_and_encode_jobs,
    encode,
    general_customer_spending_bahaviour,
    generic_customer_spending_behaviour,
    get_merchant_risk_rolling_window,
    load_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_job_categories(filepath: str) -> dict:
    """Load job categories from a JSON file."""
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        logger.error(f"Job categories file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding job categories JSON file: {e}")
        raise


def load_pickle_model(filepath: str):
    """Load a pickle model from a file."""
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logger.error(f"Pickle file not found: {e}")
        raise
    except pickle.PickleError as e:
        logger.error(f"Error loading pickle file: {e}")
        raise


def preprocess_data():
    logger.info("Starting data preprocessing pipeline.")

    # Load raw data
    test_df = load_data()
    logger.info("Data loaded successfully.")

    # Encode gender
    test_df, _ = encode(test_df, "gender", "gender", encoding="onehot")
    logger.info("Gender encoded successfully.")

    # Load and apply job categorization
    job_categories = load_job_categories("./jobs_by_category.json")
    test_df = categorize_and_encode_jobs(
        test_df, "job", job_categories, "saved_model/encoders/job_encoder.pkl"
    )
    logger.info("Jobs categorized and encoded successfully.")

    # Load and apply merchant category encoder
    merchant_cat_encoder = load_pickle_model(
        "saved_model/encoders/merchant_cat_encoder.pkl"
    )
    test_df["category_encoded"] = merchant_cat_encoder.transform(test_df[["category"]])
    logger.info("Merchant categories encoded successfully.")

    # Assign age group based on age bins
    age_bins = [0, 25, 45, 65, float("inf")]
    age_labels = [1, 0, 2, 3]
    test_df = assign_age_group(
        test_df, "dob", "trans_date_trans_time", age_bins, age_labels
    )
    logger.info("Age groups assigned successfully.")

    # Calculate distances
    test_df["distance"] = test_df.apply(calculate_distance, axis=1)
    logger.info("Geodesic distances calculated.")

    # Calculate customer spending behavior
    test_df = generic_customer_spending_behaviour(
        test_df, window_lengths=[1, 7, 30, 90, 180]
    )
    test_df = general_customer_spending_bahaviour(test_df)
    logger.info("Customer spending behavior calculated.")

    # Identify high-risk hours
    test_df["is_high_risk_hour"] = test_df["transaction_hour"].apply(
        lambda x: 2 if x in [22, 23] else (1 if x in [0, 1, 2, 3] else 0)
    )
    logger.info("High-risk hours identified.")

    # Categorize city size
    city_size_bins = [0, 88735, 1577385, float("inf")]
    city_size_labels = [0, 1, 2]
    test_df["city_size"] = pd.cut(
        test_df["city_pop"], bins=city_size_bins, labels=city_size_labels
    ).astype("int64")
    logger.info("City size categorized.")

    # Calculate merchant risk with rolling windows
    test_df = get_merchant_risk_rolling_window(
        transactions=test_df, delay_period=7, window_size=[1, 7, 30, 90, 180]
    )

    logger.info("overwriting merchant risk rolling windows")
    # Define risk windows
    risk_windows = [
        "merchant_risk_1_day_window",
        "merchant_risk_7_day_window",
        "merchant_risk_30_day_window",
        "merchant_risk_90_day_window",
        "merchant_risk_180_day_window",
    ]

    train = pd.read_csv("./data/processed/train_data.csv")
    # Select relevant columns from the train dataset for merging
    train_subset = train[["merchant", "lat", "long"] + risk_windows]

    # Merge test with train on 'merchant', 'lat', and 'long'
    merged = test_df.merge(
        train_subset,
        on=["merchant", "lat", "long"],
        how="left",
        suffixes=("", "_train"),
    )

    # Sum up the risk windows from the train dataset and assign them to the test DataFrame
    for window in risk_windows:
        test_df[window] = merged[window].fillna(0)  # Fill missing values with 0

    logger.info("Merchant risk rolling windows calculated.")

    # Save final processed data
    output_path = Path("./data/processed/test_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False, header=True)
    logger.info(f"Final data saved to '{output_path}'.")

    logger.info("Data preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    preprocess_data()
