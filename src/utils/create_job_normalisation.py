import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)  # Set the log level to INFO
logger = logging.getLogger(__name__)

from rapidfuzz import fuzz, process

def update_job_categories(test_jobs, json_file_path, threshold=70):
    """
    Update the job categories JSON file with new jobs from test data.

    Args:
        test_jobs (list): List of job titles from the test data.
        json_file_path (str): Path to the job_by_category.json file.
        threshold (int): Similarity score threshold for fuzzy matching.

    Returns:
        None: Updates the JSON file in place.
    """
    # Load the existing JSON data
    with open(json_file_path, 'r') as f:
        job_by_category = json.load(f)

    # Check for each job in the test dataset
    for test_job in test_jobs:
        found = False

        # First, check if the job already exists in the JSON file
        for category, jobs in job_by_category.items():
            if test_job in jobs:
                found = True
                break

        # If the job does not exist, use fuzzy matching
        if not found:
            best_match = None
            best_category = None
            best_score = 0

            for category, jobs in job_by_category.items():
                # Get the best match in the current category
                result = process.extractOne(test_job, jobs, scorer=fuzz.token_sort_ratio)
                if result:  # Check if result is not None
                    match, score, _ = result  # Extract match and score
                    if score > best_score:
                        best_match = match
                        best_category = category
                        best_score = score

            # Assign to the best-matching category if above threshold, otherwise "other"
            if best_score >= threshold and best_category:
                job_by_category[best_category].append(test_job)
            else:
                job_by_category["Other"].append(test_job)

    # Save the updated JSON file
    with open(json_file_path, 'w') as f:
        json.dump(job_by_category, f, indent=4)



if __name__ == "__main__":
    logger.info("load new data")
    test_df = pd.read_csv("./tr_fincrime_test.csv")
    test_jobs = test_df["job"].unique().tolist()
    logger.info("read job normalisation gazetteer")
    json_file_path = "utils/jobs_by_category.json"
    update_job_categories(test_jobs, json_file_path, threshold=70)
    logger.info("Job categories updated successfully.")