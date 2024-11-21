import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def categorize_jobs(df, job_col, job_to_category, default_category=None):
    """
    Categorize job titles and encode the resulting categories.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing job titles.
    - job_col: str - The column name with job titles.
    - job_to_category: dict - A dictionary mapping job titles to categories.
    - default_category: str or None - Default category for unmapped job titles (if None, leaves them as NaN).
    
    Returns:
    - pd.DataFrame - The DataFrame with categorized and encoded job titles.
    """
    # Map job titles to categories
    df['job_category'] = df[job_col].map(lambda job: next((category for category, jobs in job_to_category.items() if job in jobs), 'Other'))
    
    # Handle unmapped job titles
    if default_category is not None:
        df['job_category'].fillna(default_category, inplace=True)
        
    return df

def encode(df, col, cat_col, encoding='ordinal'):
    if encoding == 'label':
        # Label Encoding
        label_encoder = LabelEncoder()
        df[cat_col] = label_encoder.fit_transform(df[col])
        return df, label_encoder
    elif encoding == 'onehot':
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=[col], drop_first=True)
        return df, None  # No encoder object for one-hot
    elif encoding == 'ordinal':
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1)

        df[cat_col] = ordinal_encoder.fit_transform(df[[col]])
        return df, ordinal_encoder
    elif encoding == 'target':
        # Target Encoding
        raise NotImplementedError("Target encoding is not yet implemented.")
    else:
        raise ValueError("Invalid encoding type. Use 'label' or 'onehot' or 'ordinal'.")