from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import pickle

def create_data_splits(df, date_column, freq='90D', rolling_window=90, gap=31, val_duration=90):
    """
    Create rolling train-validation data splits from a time-indexed DataFrame.

    Parameters:
        df (pd.DataFrame): The input dataset.
        date_column (str): Column name for the transaction date.
        target_column (str): Column name for the target variable.
        freq (str): Frequency of split points (default is '90D').
        rolling_window (int): Number of days for the rolling window (default is 180 days).
        gap (int): Gap in days between the train and validation sets (default is 31 days).
        val_duration (int): Validation period duration in days (default is 90 days).

    Yields:
        dict: A dictionary with training and validation data splits:
            - 'train_df': Training features and target.
            - 'val_df', 'y_val': Validation features and target.
            - 'train_dates', 'val_dates': Date ranges for training and validation sets.
    """
    # Convert the date column to datetime and sort by it
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(date_column).set_index(date_column)

    start_date = df_sorted.index.min()
    end_date = df_sorted.index.max()

    for date in pd.date_range(start_date, end_date, freq=freq):
        train_start = date - pd.DateOffset(days=rolling_window)
        train_end = date - pd.DateOffset(days=1)
        val_start = date + pd.DateOffset(days=gap)
        val_end = date + pd.DateOffset(days=gap + val_duration)

        # Adjust start and end dates to stay within dataset bounds
        train_start = max(train_start, start_date)
        val_end = min(val_end, end_date)

        # if (train_end - train_start).days + 1 < rolling_window:
        #     print(f"Skipping split for date {date}: Training duration is less than {rolling_window} days.")
        # else:
        if train_start < train_end and val_start < val_end:
            train_df = df_sorted.loc[train_start:train_end]
            val_df = df_sorted.loc[val_start:val_end]


            yield {
                'train_df': train_df,
                'val_df': val_df,
                'train_dates': (train_df.index.min(), train_df.index.max()),
                'val_dates': (val_df.index.min(), val_df.index.max())
            }
        else:
            print(f"Skipping split for date {date}: Not enough data for train or validation set.")


def scaleData(train_df, val_df, features):
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features]) 
    return train_df, val_df

def fit_model_and_get_predictions(classifier, train_df, val_df, 
                                  input_features, save_name, output_feature="is_fraud", scale=False):
    """Trains a classifier, makes predictions, and returns the results.

    Args:
        classifier: The classifier to be trained.
        train_df: The training dataframe.
        val_df: The validation dataframe.
        input_features: A list of input feature names.
        output_feature: The name of the output feature (default: "TX_FRAUD").
        scale: Whether to scale the input features (default: True).

    Returns:
        A dictionary containing the trained classifier, predictions, and execution times.
        Raises ValueError if input dataframes are missing required columns.
    """

    # Input validation
    if not all(feature in train_df.columns for feature in input_features):
        raise ValueError("Not all input features are present in the training dataframe.")
    if output_feature not in train_df.columns:
        raise ValueError("Output feature is not present in the training dataframe.")
    if not all(feature in val_df.columns for feature in input_features):
        raise ValueError("Not all input features are present in the validation dataframe.")
    if output_feature not in val_df.columns:
        raise ValueError("Output feature is not present in the validation dataframe.")


    if scale:
        train_df, val_df = scaleData(train_df.copy(), val_df.copy(), input_features)  # Create copies to avoid modifying original dataframes

    start_time = time.time()
    classifier.fit(train_df[input_features], train_df[output_feature])
    training_execution_time = time.time() - start_time

    start_time = time.time()
    predictions_val = classifier.predict_proba(val_df[input_features])[:, 1]  
    prediction_execution_time = time.time() - start_time

    # save
    with open("./saved_model/classifier/" + str(save_name) + ".pkl", "wb") as f:
        pickle.dump(classifier, f)

    return {'classifier': classifier,
            'predictions_val': predictions_val,  
            'training_execution_time': training_execution_time,
            'prediction_execution_time': prediction_execution_time}

