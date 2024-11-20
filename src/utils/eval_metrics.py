from sklearn import metrics
import numpy as np
import pandas as pd

def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=False):
    """
    Computes the precision at top k for credit card fraud detection on a daily basis.
    Parameters:
    predictions_df (pd.DataFrame): DataFrame containing transaction predictions with columns 'trans_date', 'predictions', 'cc_num', and 'is_fraud'.
    top_k (int): The number of top predictions to consider for computing precision.
    remove_detected_compromised_cards (bool): If True, removes detected compromised cards from future days' transactions. Default is True.
    Returns:
    tuple: A tuple containing:
        - num_compromised_cards_per_day (list): List of the number of unique compromised cards per day.
        - card_precision_top_k_per_day_list (list): List of precision at top k values per day.
        - mean_card_precision_top_k (float): Mean precision at top k over all days.
    """

    # Sort days by increasing order
    list_days=list(predictions_df['trans_date'].unique())
    list_days.sort()
    
    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []
    
    card_precision_top_k_per_day_list = []
    num_compromised_cards_per_day = []
    
    # For each day, compute precision top k
    for day in list_days:
        
        df_day = predictions_df[predictions_df['trans_date']==day]
        df_day = df_day[['predictions', 'cc_num', 'is_fraud']]
        
        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[df_day['cc_num'].isin(list_detected_compromised_cards)==False]
        
        num_compromised_cards_per_day.append(len(df_day[df_day['is_fraud']==1]['cc_num'].unique()))
        
        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day,top_k)
        
        card_precision_top_k_per_day_list.append(card_precision_top_k)
        
        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)
        
    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()
    
    # Returns precision top k per day as a list, and resulting mean
    return num_compromised_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k

def performance_assessment(predictions_df, output_feature='is_fraud', 
                           prediction_feature='predictions', top_k_list=[50],
                           rounded=True):
    """
    Assess the performance of a fraud detection model using various metrics.
    Parameters:
    predictions_df (pd.DataFrame): val_df containing the true labels and predicted scores.
    output_feature (str): Column name for the true labels in predictions_df. Default is 'is_fraud'.
    prediction_feature (str): Column name for the predicted scores in predictions_df. Default is 'predictions'.
    top_k_list (list of int): List of top-k values for which to calculate card precision. Default is [100].
    rounded (bool): Whether to round the performance metrics to 3 decimal places. Default is True.
    Returns:
    pd.DataFrame: DataFrame containing the calculated performance metrics including AUC ROC, Average Precision, 
        and Card Precision for each top-k value in top_k_list.
    """
    auc_roc = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    ap = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
    f1 = metrics.f1_score(predictions_df[output_feature], predictions_df[prediction_feature].round())
    performances = pd.DataFrame([[auc_roc, ap, f1]], 
                           columns=['auc_roc', 'Average precision', 'F1 score'])
    
    # for top_k in top_k_list:
    
    #     _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
    #     performances['Card Precision@'+str(top_k)]=mean_card_precision_top_k
        
    if rounded:
        performances = performances.round(3)
    
    return performances