import logging
import os
import pickle
import sys
import time

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from data.data_transformation import scaleData
from utils.eval_metrics import precision_top_k


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure relative imports work
sys.path.append(os.path.abspath(os.path.join(".", "src")))


def fit_model(
    classifier,
    train_df,
    input_features,
    output_feature="is_fraud",
    scale=False,
    save=False,
    save_model_name=None,
):
    """
    Trains a classifier on the provided data.

    Args:
        classifier (object): Pre-initialized model.
        train_df (pd.DataFrame): Training data.
        input_features (list): Feature columns for training.
        output_feature (str): Target column name (default: "is_fraud").
        scale (bool): Whether to scale input features (default: False).
        save (bool): Whether to save the trained model (default: False).
        save_model_name (str): Name for saving the model (if save=True).

    Returns:
        object: Trained classifier.
    """
    if not all(feature in train_df.columns for feature in input_features):
        raise ValueError(
            "Not all input features are present in the training dataframe."
        )
    if output_feature not in train_df.columns:
        raise ValueError("Output feature is not present in the training dataframe.")

    if scale:
        train_df = scaleData(train_df.copy(), input_features)

    logger.info(f"Training ...")
    start_time = time.time()
    classifier.fit(train_df[input_features], train_df[output_feature])
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")

    if save and save_model_name:
        save_path = os.path.join(
            "..", "saved_model", "classifier", f"{save_model_name}.pkl"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(classifier, f)
        logger.info(f"Model saved to {save_path}.")

    return classifier


def predict(classifier, train_df, val_df, input_features, output_feature="is_fraud"):
    """
    Generates predictions for training and validation sets.

    Args:
        classifier (object): Trained classifier.
        train_df, val_df (pd.DataFrame): Datasets for training and validation.
        input_features (list): Feature columns for prediction.
        output_feature (str): Target column name.

    Returns:
        dict: Predictions for training and validation sets.
    """
    if not all(feature in val_df.columns for feature in input_features):
        raise ValueError(
            "Not all input features are present in the validation dataframe."
        )
    if output_feature not in val_df.columns:
        raise ValueError("Output feature is not present in the validation dataframe.")

    logger.info("Generating predictions...")
    start_time = time.time()
    predictions_train = classifier.predict_proba(train_df[input_features])[:, 1]
    predictions_val = classifier.predict_proba(val_df[input_features])[:, 1]
    prediction_time = time.time() - start_time
    logger.info(f"Predictions completed in {prediction_time:.2f} seconds.")

    return {"predictions_train": predictions_train, "predictions_val": predictions_val}


def evaluate_model_on_splits(
    splits, classifier, input_features, top_k_list=[100], summary="Default"
):
    """
    Evaluates a classifier across multiple data splits.

    Args:
        splits (list): List of splits, each containing `train_df`, `val_df`, `train_dates`, and `val_dates`.
        classifier (object): Pre-initialized classifier.
        input_features (list): Feature columns for training and evaluation.
        top_k_list (list): List of top-k values for evaluation metrics.
        summary (str): Summary of model parameters.

    Returns:
        tuple: Aggregated performance metrics and metrics per split.
    """
    performances_df_folds = pd.DataFrame()
    start_time = time.time()

    for split in splits:
        logger.info(f"Train dates: {split['train_dates']}")
        logger.info(f"Validation dates: {split['val_dates']}")
        train_df, val_df = split["train_df"], split["val_df"]

        clf = fit_model(classifier, train_df, input_features)
        predictions = predict(clf, train_df, val_df, input_features)

        train_df["predictions"] = predictions["predictions_train"]
        val_df["predictions"] = predictions["predictions_val"]

        perf_val = performance_assessment(val_df, top_k_list=top_k_list, rounded=False)
        perf_val.columns += " val"
        perf_train = performance_assessment(
            train_df, top_k_list=top_k_list, rounded=False
        )
        perf_train.columns += " train"

        performances_df_folds = pd.concat(
            [performances_df_folds, pd.concat([perf_val, perf_train], axis=1)],
            ignore_index=True,
        )

    execution_time = time.time() - start_time
    perf_mean = performances_df_folds.mean().to_frame().T
    perf_std = performances_df_folds.std(ddof=0).to_frame().T
    perf_std.columns += " Std"

    performances_df = pd.concat([perf_mean, perf_std], axis=1)
    performances_df["Execution time"] = execution_time
    performances_df["Parameters summary"] = summary

    return performances_df, performances_df_folds


def performance_assessment(
    predictions_df,
    output_feature="is_fraud",
    prediction_feature="predictions",
    top_k_list=[50],
    rounded=True,
):
    """
    Calculates performance metrics for fraud detection.

    Args:
        predictions_df (pd.DataFrame): Dataframe with true labels and predicted scores.
        output_feature (str): Column name for true labels.
        prediction_feature (str): Column name for predicted scores.
        top_k_list (list): List of top-k values for precision calculation.
        rounded (bool): Whether to round metrics.

    Returns:
        pd.DataFrame: Dataframe with calculated metrics.
    """
    auc_roc = metrics.roc_auc_score(
        predictions_df[output_feature], predictions_df[prediction_feature]
    )
    ap = metrics.average_precision_score(
        predictions_df[output_feature], predictions_df[prediction_feature]
    )
    f1 = metrics.f1_score(
        predictions_df[output_feature], predictions_df[prediction_feature].round()
    )

    performances = pd.DataFrame(
        [[auc_roc, ap, f1]], columns=["auc_roc", "average precision", "f1 score"]
    )

    for top_k in top_k_list:
        _, _, mean_precision_top_k = precision_top_k(predictions_df, top_k)
        performances[f"precision_top_{top_k}"] = mean_precision_top_k

    return performances.round(3) if rounded else performances
