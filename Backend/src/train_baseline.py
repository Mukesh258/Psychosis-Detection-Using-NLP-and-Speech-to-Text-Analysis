"""
Train baseline models: Logistic Regression and XGBoost with TF-IDF features.
"""
import argparse
import pickle
import os
from pathlib import Path
import numpy as np  # type: ignore
# sklearn imports - ensure scikit-learn is installed: pip install scikit-learn
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
import xgboost as xgb  # type: ignore

from data_loader import load_dataset, get_data_splits
from preprocess import preprocess_batch
from features import create_tfidf_vectorizer


def train_logistic_regression(X_train, y_train, X_test, y_test, max_features=5000):
    """Train Logistic Regression model with TF-IDF."""
    print("Training Logistic Regression...")
    
    # Create pipeline
    vectorizer = create_tfidf_vectorizer(max_features=max_features)
    model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline


def train_xgboost(X_train, y_train, X_test, y_test, max_features=5000):
    """Train XGBoost model with TF-IDF."""
    print("\nTraining XGBoost...")
    
    # Vectorize
    vectorizer = create_tfidf_vectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder  # type: ignore
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train model
    model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss'
    )
    
    model.fit(X_train_vec, y_train_encoded)
    
    # Evaluate
    y_pred_encoded = model.predict(X_test_vec)
    y_pred = le.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return both model and vectorizer
    return {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': le
    }


def save_model(model, model_path: str, vectorizer=None, label_encoder=None):
    """Save model and associated components."""
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if vectorizer is None and label_encoder is None:
        # Pipeline model (Logistic Regression)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    else:
        # XGBoost model (separate components)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'vectorizer': vectorizer,
                'label_encoder': label_encoder
            }, f)
        print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--max_features', type=int, default=5000,
                       help='Maximum TF-IDF features')
    
    args = parser.parse_args()
    
    # Resolve data path - handle relative paths properly
    data_path = Path(args.data)
    if not data_path.is_absolute():
        # If relative path doesn't exist, try relative to script directory
        if not data_path.exists():
            script_dir = Path(__file__).parent
            alt_path = script_dir.parent / data_path
            if alt_path.exists():
                data_path = alt_path
            else:
                # Try relative to current working directory's parent
                alt_path = Path.cwd().parent / data_path
                if alt_path.exists():
                    data_path = alt_path
    
    # Convert to absolute path
    data_path = data_path.resolve()
    
    # Load and preprocess data
    print(f"Loading dataset from {data_path}...")
    df = load_dataset(str(data_path))
    print(f"Loaded {len(df)} samples")
    
    # Preprocess
    df['text'] = preprocess_batch(df['text'].values)
    
    # Split data
    X_train, X_test, y_train, y_test = get_data_splits(df)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create models directory
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, X_test, y_test, 
                                        max_features=args.max_features)
    save_model(lr_model, str(models_dir / "logistic_regression.pkl"))
    
    # Train XGBoost
    xgb_components = train_xgboost(X_train, y_train, X_test, y_test,
                                   max_features=args.max_features)
    save_model(
        xgb_components['model'],
        str(models_dir / "xgboost.pkl"),
        vectorizer=xgb_components['vectorizer'],
        label_encoder=xgb_components['label_encoder']
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

