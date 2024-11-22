import pandas as pd
from sklearn.classifier import RandomForestClassifier


def get_feature_importance(df, input_feature, top_features):
    """
    Calculate and return the importance of features using a RandomForestClassifier.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.
    input_feature (list): List of feature names to be used for training the classifier.
    top_features (int): Number of top features to return based on their importance.
    Returns:
    pd.DataFrame: A DataFrame containing the top features and their importance scores, sorted by importance in descending order.
    """
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(df[input_feature], df["is_fraud"])
    feature_importances = clf.feature_importances_
    feature_importances = pd.DataFrame(
        {"feature": input_feature, "importance": feature_importances}
    )
    feature_importances = feature_importances.sort_values(
        "importance", ascending=False
    )[:top_features]
    return feature_importances
