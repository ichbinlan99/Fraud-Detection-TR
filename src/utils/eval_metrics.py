import numpy as np


def precision_top_k_day(df_day, top_k=100):
    """
    Calculate precision for the top K transactions of a single day.

    Args:
        df_day (DataFrame): Data for a single day with fraud predictions.
        top_k (int): The number of top transactions to evaluate.
                     If None, defaults to the number of actual frauds.

    Returns:
        detected_fraudulent_transactions (list): List of detected fraudulent transaction IDs.
        precision_top_k (float): Precision for the top K transactions.
    """
    if top_k is None:
        top_k = df_day[
            "is_fraud"
        ].sum()  # Default to the number of fraud cases if not specified.

    # Order transactions by descending fraud prediction probabilities.
    df_day = df_day.sort_values(by="predictions", ascending=False).reset_index(
        drop=True
    )

    # Select the top K most suspicious transactions.
    df_day_top_k = df_day.head(top_k)

    # Identify detected fraudulent transactions.
    detected_fraudulent_transactions = df_day_top_k[df_day_top_k["is_fraud"] == 1][
        "trans_num"
    ].tolist()

    # Compute precision as the ratio of correctly identified frauds to top K.
    precision_top_k = len(detected_fraudulent_transactions) / top_k

    return detected_fraudulent_transactions, precision_top_k


def precision_top_k(predictions_df, top_k=100):
    """
    Compute the Top K Precision metric over multiple days.

    Args:
        predictions_df (DataFrame): Data containing fraud predictions and transaction details.
        top_k (int): The number of top transactions to evaluate per day.

    Returns:
        num_fraudulent_transactions_per_day (list): Fraud counts per day.
        precision_top_k_per_day_list (list): Precision values for each day.
        mean_precision_top_k (float): Mean precision across all days.
    """
    # Extract unique transaction dates.
    unique_days = sorted(predictions_df["trans_date"].unique())

    precision_top_k_per_day_list = []
    num_fraudulent_transactions_per_day = []

    # Compute precision for each day.
    for day in unique_days:
        df_day = predictions_df[predictions_df["trans_date"] == day][
            ["trans_num", "is_fraud", "predictions"]
        ]

        # Count the number of fraudulent transactions for the day.
        fraud_count = df_day["is_fraud"].sum()
        num_fraudulent_transactions_per_day.append(fraud_count)

        # Compute the precision for the day.
        _, day_precision = precision_top_k_day(df_day, top_k=top_k)
        precision_top_k_per_day_list.append(day_precision)

    # Calculate the mean precision across all days.
    mean_precision_top_k = np.round(np.mean(precision_top_k_per_day_list), 3)

    return (
        num_fraudulent_transactions_per_day,
        precision_top_k_per_day_list,
        mean_precision_top_k,
    )
