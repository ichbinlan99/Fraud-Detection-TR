from datetime import datetime, timedelta
import pandas as pd

def get_train_val_set(transactions_df, start_date_training, delta_train=30, delta_delay=14, delta_val=30):
    """
    Splits the transactions DataFrame into training and validation sets based on the provided dates and intervals.
    Parameters:
    transactions_df (pd.DataFrame): DataFrame containing transaction data with 'trans_date_trans_time' and 'is_fraud' columns.
    start_date_training (str or pd.Timestamp): The start date for the training period.
    delta_train (int, optional): Number of days for the training period. Default is 30.
    delta_delay (int, optional): Number of days between the end of the training period and the start of the validation period. Default is 14.
    delta_val (int, optional): Number of days for the validation period. Default is 30.
    Returns:
    tuple: A tuple containing two DataFrames:
        - train_df (pd.DataFrame): DataFrame containing the training set transactions.
        - val_df (pd.DataFrame): DataFrame containing the validation set transactions, excluding known fraudulent transactions.
    """
    
    # Convert dates to datetime objects upfront for efficiency
    start_date_training = pd.to_datetime(start_date_training)
    transactions_df['trans_date_trans_time'] = pd.to_datetime(transactions_df['trans_date_trans_time'])
    transactions_df['trans_date'] = transactions_df['trans_date_trans_time'].dt.date

    # Calculate end date for training data once
    end_date_training = start_date_training + timedelta(days=delta_train)

    # Get training set using a single boolean mask
    train_df = transactions_df[
        (transactions_df['trans_date_trans_time'] >= start_date_training) &
        (transactions_df['trans_date_trans_time'] < end_date_training)
    ]

    # Pre-calculate start and end dates for validation period
    start_date_val = start_date_training + timedelta(days=delta_train + delta_delay)
    end_date_val = start_date_val + timedelta(days=delta_val)

    # Filter transactions within the validation period once
    val_df_all = transactions_df[
        (transactions_df['trans_date_trans_time'] >= start_date_val) &
        (transactions_df['trans_date_trans_time'] < end_date_val)
    ]
    
    # Get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df['is_fraud']==1]['cc_num'])

    val_dfs = []  # List to store daily validation DataFrames
    current_date = start_date_val

    for day in range(delta_val):
        # Efficiently filter for the current day
        val_df_day = val_df_all[val_df_all['trans_date'] == current_date.date()]
        
        # Calculate the delay period date only once per loop
        delay_date = start_date_training + timedelta(days=delta_train + day -1)
        val_df_day_delay_period = transactions_df[transactions_df['trans_date'] == delay_date.date()]


        # Update known defrauded customers
        new_defrauded_customers = set(val_df_day_delay_period[val_df_day_delay_period['is_fraud']==1]['cc_num'])
        known_defrauded_customers.update(new_defrauded_customers) # Use update for better performance

        # Filter out known fraudulent transactions
        val_df_day = val_df_day[~val_df_day['cc_num'].isin(known_defrauded_customers)]
        val_dfs.append(val_df_day)

        current_date += timedelta(days=1)

    val_df = pd.concat(val_dfs)

    return train_df, val_df


from sklearn.preprocessing import StandardScaler
import time

def scaleData(train_df, val_df, features):
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features]) 
    return train_df, val_df

def fit_model_and_get_predictions(classifier, train_df, val_df, 
                                  input_features, output_feature="is_fraud", scale=False):
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

    predictions_train = classifier.predict_proba(train_df[input_features])[:, 1]

    return {'classifier': classifier,
            'predictions_val': predictions_val,  
            'predictions_train': predictions_train,
            'training_execution_time': training_execution_time,
            'prediction_execution_time': prediction_execution_time}

