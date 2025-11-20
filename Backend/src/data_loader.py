"""
Data loading utilities for the psychosis detection dataset.
"""
import pandas as pd  # type: ignore
import os
from pathlib import Path


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the synthetic psychosis dataset from CSV.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Dataset must contain 'text' and 'label' columns. Found columns: {list(df.columns)}")
    
    # Drop rows with missing text or label
    before = len(df)
    df = df.dropna(subset=['text', 'label'])
    after = len(df)
    if after < before:
        print(f"Warning: Dropped {before - after} rows with missing text/label")
    
    # Normalize labels to a consistent, multi-class scheme
    # e.g. normal, anxiety, stress, depression, suicidal, bipolar, personality disorder, psychotic-like, etc.
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    
    # Canonical mapping for common variants
    label_mapping = {
        'personality disorder': 'personality disorder',
        'personality_disorder': 'personality disorder',
        'personalitydisorder': 'personality disorder',
        'bipolar disorder': 'bipolar',
        'bipolar_disorder': 'bipolar',
        'psychotic_like': 'psychotic-like',
        'psychoticlike': 'psychotic-like',
    }
    df['label'] = df['label'].replace(label_mapping)
    
    # For readability in downstream UI, keep labels in a title-cased form
    df['label'] = df['label'].str.replace('_', ' ').str.title()
    
    return df


def get_data_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        (X_train, X_test, y_train, y_test) tuple
    """
    from sklearn.model_selection import train_test_split  # type: ignore
    
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test data loading
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "synthetic_psychosis_data.csv"
    
    df = load_dataset(str(data_path))
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nSample text:\n{df.iloc[0]['text']}")

