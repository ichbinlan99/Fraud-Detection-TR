from datetime import datetime, timedelta
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)  # Set the log level to INFO
logger = logging.getLogger(__name__)


def create_data_splits(
    df, date_column, freq="90D", rolling_window=90, gap=31, val_duration=90
):
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
            - 'train_df', 'val_df': Training and validation DataFrames.
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

        if (train_end - train_start).days + 1 < rolling_window:
            logger.info(
                f"Skipping split for date {date}: Training duration is less than {rolling_window} days."
            )
        elif (val_end - val_start).days + 1 < val_duration:
            logger.info(
                f"Skipping split for date {date}: Validation duration is less than {val_duration} days."
            )
        else:
            if train_start < train_end and val_start < val_end:
                train_df = df_sorted.loc[train_start:train_end]
                val_df = df_sorted.loc[val_start:val_end]

                yield {
                    "train_df": train_df,
                    "val_df": val_df,
                    "train_dates": (train_df.index.min(), train_df.index.max()),
                    "val_dates": (val_df.index.min(), val_df.index.max()),
                }
            else:
                print(
                    f"Skipping split for date {date}: Not enough data for train or validation set."
                )
