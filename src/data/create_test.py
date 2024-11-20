import warnings
import pandas as pd
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from geopy.distance import geodesic
from preprocessing import categorize_jobs, encode, customer_spending_behaviour, get_merchant_risk_rolling_window
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Load datasets
def load_data():
    """
    Load train and test datasets.
    """
    logger.info("Loading datasets...")
    test_df = pd.read_csv('tr_fincrime_test.csv')
    logger.info(f"Loaded test and train datasets with shapes {test_df.shape}.")
    return test_df

# Utility functions
def load_encoder(path):
    """
    Load saved encoder from path.
    """
    logger.info(f"Loading encoder from {path}...")
    with open(path, 'rb') as file:
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
    with open(path, 'rb') as file:
        loaded_label_encoder = pickle.load(file)
    loaded_label_encoder.classes_ = np.append(loaded_label_encoder.classes_, 'Other')
    df['job_encoded'] = loaded_label_encoder.transform(df['job_category'])
    logger.info("Jobs categorized and encoded.")
    return df

# Main preprocessing flow
def preprocess_data():
    test_df = load_data()

    # Preprocessing: Encoding gender
    test_df_enc_gender, _ = encode(test_df, "gender", "gender", encoding="onehot")

    # Load job normalization gazetteer
    with open('utils/jobs_by_category.json', 'r') as f:
        job_categories = json.load(f)

    # Categorize and encode jobs
    test_df_cat_job = categorize_and_encode_jobs(test_df_enc_gender, 'job', job_categories, 'saved_model/job_encoder.pkl')

    # Load merchant category encoder and apply
    with open("saved_model/merchant_cat_encoder.pkl", 'rb') as file:
        merchant_cat_encoder = pickle.load(file)
    new_categories = [cat for cat in test_df_cat_job['category'].unique() if cat not in merchant_cat_encoder.classes_]
    merchant_cat_encoder.classes_ = np.concatenate([merchant_cat_encoder.classes_, new_categories])
    test_df_cat_job['category_encoded'] = merchant_cat_encoder.transform(test_df_cat_job['category'])

    # Calculate age and group into age ranges
    age_bins = [0, 25, 45, 65, float("inf")]
    age_labels = [1, 0, 2, 3]  # Fraud risk labels
    test_df_cat_job = assign_age_group(test_df_cat_job, "dob", "trans_date_trans_time", age_bins, age_labels)

    # Calculate distances
    test_df_cat_job["distance"] = test_df_cat_job.apply(calculate_distance, axis=1)
    logger.info("Geodesic distance calculated for each transaction.")

    # Customer spending behavior
    test_df_dist = customer_spending_behaviour(test_df_cat_job, window_lengths=[30, 90, 180])
    logger.info("Customer spending behavior calculated.")

    # Identify high-risk hours
    test_df_dist["is_high_risk_hour"] = test_df_dist["transaction_hour"].apply(
        lambda x: 2 if x in [22, 23] else (1 if x in [0, 1, 2, 3] else 0)
    )
    logger.info("High-risk hours identified.")

    # City size categorization
    city_size_bins = [0, 88735, 1577385, float("inf")]
    city_size_labels = [0, 1, 2]
    test_df_dist["city_size"] = pd.cut(test_df_dist["city_pop"], bins=city_size_bins, labels=city_size_labels).astype("int64")
    logger.info("City size categorization completed.")

    # Merchant risk rolling window
    test_df_merch_risk = get_merchant_risk_rolling_window(transactions=test_df_dist, delay_period=7, window_size=[1, 7, 30])
    logger.info("Merchant risk rolling window calculated.")

    # Save final DataFrame to CSV
    test_df_merch_risk.to_csv("data/test_data.csv", index=False, header=True)
    logger.info("Final data saved to 'data/test_data.csv'.")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data()

