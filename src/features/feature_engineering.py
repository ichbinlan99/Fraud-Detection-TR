import pandas as pd
import holidays

def generic_customer_spending_behaviour(df, window_lengths=[7, 30, 90, 180]):
    """
    Calculates features related to customer spending behavior, including:
    - avg_distance_[window_length]_days: Rolling average distance for the past [window_length] days.
    - distance_over_avg_[window_length]_days: Distance of transaction over rolling average distance for the past [window_length] days.
    - avg_amount_[window_length]_days: Rolling average amount for the past [window_length] days.
    - count_amount_[window_length]_days: Number of transactions for the past [window_length] days.
    - amount_over_average_[window_length]_days: Amount of transaction over rolling average amount for the past [window_length] days.
    - transaction_hour: Hour of the transaction.
    - transaction_day_of_week: Day of the week of the transaction (0-6, Monday-Sunday).
    - is_holiday: Whether the transaction occurred on a US public holiday.
    - is_weekend: Whether the transaction occurred on a weekend.
    - daily_trans_count: Number of transactions for the same credit card on the same day.

    Args:
        df: The input DataFrame.
        window_lengths: A list of window lengths (in days) for calculating rolling averages.

    Returns:
        DataFrame with added features.
    """

    # --- Distance-based features ---
    for window_length in window_lengths:
        # -- distance features --
        # Calculate transaction rolling average distance
        df[f"avg_distance_{window_length}_days"] = df.sort_values(by=["trans_date_trans_time"]).groupby("cc_num").apply(
            lambda x: x.rolling(f'{window_length}D', on='trans_date_trans_time')["distance"].mean().shift(1)
        ).reset_index(0, drop=True)
        # Calculate average distance per city
        avg_distance_per_city = df.groupby("city")["distance"].mean()
        # Fill NaN values in rolling average with city average
        df[f"avg_distance_{window_length}_days"] = df.apply(
            lambda row: avg_distance_per_city[row["city"]] if pd.isna(row[f"avg_distance_{window_length}_days"]) else row[f"avg_distance_{window_length}_days"],
            axis=1
        )
        # Calculate distance over average distance
        df[f"distance_over_avg_{window_length}_days"] = df["distance"] / df[f"avg_distance_{window_length}_days"]

        # -- amount features --
        # Calculate transaction rolling average amount
        df[f"sum_amount_{window_length}_days"] = df.sort_values(by=["trans_date_trans_time"]).groupby("cc_num").apply(
            lambda x: x.rolling(f'{window_length}D', on='trans_date_trans_time')["amt"].sum().shift(1)
        ).reset_index(level=0, drop=True)
        df[f"count_amount_{window_length}_days"] = df.sort_values(by=["trans_date_trans_time"]).groupby("cc_num").apply(
            lambda x: x.rolling(f'{window_length}D', on='trans_date_trans_time')["amt"].count().shift(1)
        ).reset_index(level=0, drop=True)
        df[f"avg_amount_{window_length}_days"] = df[f"sum_amount_{window_length}_days"] / df[f"count_amount_{window_length}_days"]

        # Calculate average amount per city and fill NaN values in rolling average with city average
        city_average = df.groupby("city")["amt"].mean()
        df[f"avg_amount_{window_length}_days"] = df[f"avg_amount_{window_length}_days"].fillna(df["city"].map(city_average))

        # Calculate amount over average
        df[f"amount_over_average_{window_length}_days"] = df["amt"] / df[f"avg_amount_{window_length}_days"]

        # Calculate inter-transaction time and fill NaN values in inter-transaction time with the mean inter-transaction time
        df[f"inter_transaction_time_{window_length}_days"] = df.sort_values(by=["trans_date_trans_time"]).groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds().shift()
        df[f"inter_transaction_time_{window_length}_days"] = df[f"inter_transaction_time_{window_length}_days"].fillna(df[f"inter_transaction_time_{window_length}_days"].mean())

        #Calculate the number of unique days
        df["trans_date"] = df["trans_date_trans_time"].dt.date

        # Calculate transaction counts
        card_counts = df.groupby('cc_num')['amt'].count() 
        card_amounts = df.groupby('cc_num')['amt'].sum()
        merchant_counts = df.groupby('merchant')['amt'].count() 
        merchant_amounts = df.groupby('merchant')['amt'].sum() 

        # #Calculate sparsity as average transaction counts per card per day
        cards_count_sparsity = len(df['trans_date'].unique())/ card_counts
        merchants_count_sparsity = len(df['trans_date'].unique())/ merchant_counts

        # #Calculate sparsity as average transaction amounts (per card) per day
        cards_amount_sparsity = len(df['trans_date'].unique())/ card_amounts
        merchants_amount_sparsity = len(df['trans_date'].unique())/ merchant_amounts
        df['card_count_sparsity'] = df['cc_num'].map(cards_count_sparsity)
        df['merchant_count_sparsity'] = df['merchant'].map(merchants_count_sparsity)
        df['card_amount_sparsity'] = df['cc_num'].map(cards_amount_sparsity)
        df['merchant_amount_sparsity'] = df['merchant'].map(merchants_amount_sparsity)
    return df

def general_customer_bahaviour(df):

    # --- Time-based features ---
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    us_holidays = holidays.US()
    df["is_holiday"] = df["trans_date_trans_time"].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    df["is_weekend"] = df["transaction_day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # --- Transaction count features ---
    df["trans_date"] = df["trans_date_trans_time"].dt.date
    df["daily_trans_count"] = df.groupby(["cc_num", "trans_date"])["trans_num"].transform("count")

    # --- Overal sparsity features ---
    # Calculate transaction counts
    card_counts = df.groupby('cc_num')['amt'].count() 
    card_amounts = df.groupby('cc_num')['amt'].sum()
    merchant_counts = df.groupby('merchant')['amt'].count() 
    merchant_amounts = df.groupby('merchant')['amt'].sum() 

    return df


def get_merchant_risk_rolling_window(transactions, delay_period=7, window_size=[1, 7, 30]):
    """
    Computes risk scores for merchants based on rolling windows of transactions.

    Args:
        transactions: DataFrame of transactions. Must contain 'trans_date_trans_time', 'merchant', and 'is_fraud' columns.
        delay_period: Delay period in days.
        window_size: List of window sizes in days.

    Returns:
        DataFrame with added risk score and transaction count features.
    """

    num_fraud_delay = transactions.sort_values(by=["trans_date_trans_time"]).groupby("merchant").apply(
            lambda x: x.rolling(f'{delay_period}D', on='trans_date_trans_time')['is_fraud'].sum()
            ).reset_index(level=0, drop=True)
    num_trans_delay = transactions.sort_values(by=["trans_date_trans_time"]).groupby("merchant").apply(
            lambda x: x.rolling(f'{delay_period}D', on='trans_date_trans_time')['is_fraud'].count()
            ).reset_index(level=0, drop=True)

    for window_size in window_size:
        num_fraud_delay_window = transactions.sort_values(by=["trans_date_trans_time"]).groupby("merchant").apply(
            lambda x: x.rolling(f'{delay_period + window_size}D', on='trans_date_trans_time')['is_fraud'].sum()
            ).reset_index(level=0, drop=True)
        num_trans_delay_window = transactions.sort_values(by=["trans_date_trans_time"]).groupby("merchant").apply(
            lambda x: x.rolling(f'{delay_period + window_size}D', on='trans_date_trans_time')['is_fraud'].count()
            ).reset_index(level=0, drop=True)

        num_fraud_window = num_fraud_delay_window - num_fraud_delay
        num_trans_window = num_trans_delay_window - num_trans_delay

        risk_window = num_fraud_window / num_trans_window

        transactions['merchant_num_trans_' + str(window_size) + '_day_window'] = num_trans_window.values
        transactions['merchant_risk_' + str(window_size) + '_day_window'] = risk_window.values

    transactions.fillna(0, inplace=True)  # Replace NA with 0

    return transactions

