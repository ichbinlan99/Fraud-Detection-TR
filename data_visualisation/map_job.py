import pandas as pd
from sklearn.preprocessing import LabelEncoder

def categorize_and_encode_jobs(df, job_col, job_to_category, encoding='label', default_category=None):
    """
    Categorize job titles and encode the resulting categories.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing job titles.
    - job_col: str - The column name with job titles.
    - job_to_category: dict - A dictionary mapping job titles to categories.
    - encoding: str - Encoding type: 'label' for Label Encoding, 'onehot' for One-Hot Encoding.
    - default_category: str or None - Default category for unmapped job titles (if None, leaves them as NaN).
    
    Returns:
    - pd.DataFrame - The DataFrame with categorized and encoded job titles.
    """
    # Step 1: Map job titles to categories
    df['job_category'] = df["job"].map(lambda job: next((category for category, jobs in job_to_category.items() if job in jobs), 'Unknown'))
    
    # Step 2: Handle unmapped job titles
    if default_category is not None:
        df['job_category'].fillna(default_category, inplace=True)

    # Step 3: Encode the categories
    if encoding == 'label':
        # Label Encoding
        label_encoder = LabelEncoder()
        df['category_encoded'] = label_encoder.fit_transform(df['job_category'])
        return df
    elif encoding == 'onehot':
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['job_category'])
        return df
    else:
        raise ValueError("Invalid encoding type. Use 'label' or 'onehot'.")


def encode_category(df, col, cat_col):
    label_encoder = LabelEncoder()
    df['cat_col'] = label_encoder.fit_transform(df['col'])
    return df