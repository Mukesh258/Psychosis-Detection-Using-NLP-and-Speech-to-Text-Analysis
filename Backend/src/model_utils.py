"""
Model loading utilities.
"""
import pickle
import os
from pathlib import Path
import numpy as np  # type: ignore
import torch  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification  # type: ignore


def load_vectorizer(model_path: str):
    """Load TF-IDF vectorizer from saved model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Check if it's a pipeline or separate components
    if isinstance(model_data, Pipeline):
        return model_data.named_steps['tfidf']
    elif isinstance(model_data, dict) and 'vectorizer' in model_data:
        return model_data['vectorizer']
    else:
        raise ValueError("Could not extract vectorizer from model")


def load_baseline_model(model_path: str):
    """Load baseline model (Logistic Regression or XGBoost)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Pipeline model (Logistic Regression)
    if isinstance(model_data, Pipeline):
        return {
            'model': model_data,
            'type': 'pipeline',
            'label_encoder': None
        }
    
    # Separate components (XGBoost)
    elif isinstance(model_data, dict):
        return {
            'model': model_data['model'],
            'vectorizer': model_data['vectorizer'],
            'label_encoder': model_data.get('label_encoder', None),
            'type': 'xgboost'
        }
    
    else:
        raise ValueError(f"Unknown model format in {model_path}")


def load_bert_model(model_dir: str):
    """Load fine-tuned DistilBERT model and tokenizer."""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at {model_dir}")
    
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()  # Set to evaluation mode
    
    # Load label encoder if available
    label_encoder_path = model_dir.parent / "distilbert_label_encoder.pkl"
    label_encoder = None
    if label_encoder_path.exists():
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder
    }


def predict_text(model_data: dict, text: str, model_type: str = 'baseline'):
    """
    Predict label and probability for a text using loaded model.
    
    Args:
        model_data: Dictionary containing model components
        text: Input text string
        model_type: 'baseline' or 'bert'
        
    Returns:
        Dictionary with 'label', 'prob', 'tokens', 'token_importances'
    """
    if model_type == 'baseline':
        return _predict_baseline(model_data, text)
    elif model_type == 'bert':
        return _predict_bert(model_data, text)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _predict_baseline(model_data: dict, text: str):
    """Predict using baseline model."""
    model = model_data['model']
    model_type = model_data['type']
    
    # Get vectorizer
    if model_type == 'pipeline':
        vectorizer = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        label_encoder = None
    else:  # xgboost
        vectorizer = model_data['vectorizer']
        clf = model_data['model']
        label_encoder = model_data.get('label_encoder')
    
    # Vectorize text
    X = vectorizer.transform([text])
    
    # Predict
    proba = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]
    
    # Get label
    if label_encoder:
        label = label_encoder.inverse_transform([pred])[0]
    else:
        label = pred
    
    # Get token importances (top features)
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0] if hasattr(clf, 'coef_') else clf.feature_importances_
    
    # Get token-level importance (simplified - using TF-IDF values)
    tokens = vectorizer.transform([text])
    token_scores = tokens.toarray()[0]
    
    # Get top tokens
    top_indices = np.argsort(np.abs(token_scores * coef))[-20:][::-1]
    token_list = []
    token_importances = []
    
    for idx in top_indices:
        if token_scores[idx] > 0:
            token_list.append(feature_names[idx])
            # Importance is TF-IDF value weighted by model coefficient/importance
            importance = token_scores[idx] * coef[idx]
            token_importances.append(float(importance))
    
    # Map probabilities to labels
    if label_encoder:
        classes = label_encoder.classes_
    else:
        classes = clf.classes_
    
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}
    
    return {
        'label': str(label),
        'prob': float(prob_dict.get(label, proba.max())),
        'probs': prob_dict,
        'tokens': token_list,
        'token_importances': token_importances
    }


def _predict_bert(model_data: dict, text: str):
    """Predict using DistilBERT model."""
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    label_encoder = model_data.get('label_encoder')
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    
    # Get prediction
    pred_idx = np.argmax(probs)
    prob = float(probs[pred_idx])
    
    # Get label
    if label_encoder:
        label = label_encoder.inverse_transform([pred_idx])[0]
        classes = label_encoder.classes_
    else:
        label = 'psychotic-like' if pred_idx == 1 else 'normal'
        classes = ['normal', 'psychotic-like']
    
    # Token-level importance (simplified - using attention)
    input_ids = encoding['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)
        attention = outputs.attentions[-1]  # Last layer attention
        # Average over all heads
        attention_avg = attention.mean(dim=1).squeeze(0)
        # Average over sequence dimension (sum of all tokens attending to each token)
        token_attention = attention_avg.mean(dim=0).cpu().numpy()
    
    # Filter out special tokens
    valid_tokens = []
    valid_importances = []
    for i, (token, attn) in enumerate(zip(tokens, token_attention)):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            valid_tokens.append(token.replace('##', ''))
            valid_importances.append(float(attn))
    
    # Normalize importances
    if valid_importances:
        max_imp = max(valid_importances)
        if max_imp > 0:
            valid_importances = [imp / max_imp for imp in valid_importances]
    
    # Create prob dict
    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    
    return {
        'label': str(label),
        'prob': prob,
        'probs': prob_dict,
        'tokens': valid_tokens[:50],  # Limit to 50 tokens
        'token_importances': valid_importances[:50]
    }


if __name__ == "__main__":
    # Test model loading
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    
    test_text = "I see patterns everywhere that others can't see."
    
    # Try loading baseline model
    try:
        lr_path = models_dir / "logistic_regression.pkl"
        if lr_path.exists():
            print("Loading Logistic Regression model...")
            lr_model = load_baseline_model(str(lr_path))
            result = predict_text(lr_model, test_text, 'baseline')
            print(f"Prediction: {result['label']}, Probability: {result['prob']:.4f}")
    except Exception as e:
        print(f"Could not load baseline model: {e}")
    
    # Try loading BERT model
    try:
        bert_dir = models_dir / "distilbert"
        if bert_dir.exists():
            print("\nLoading DistilBERT model...")
            bert_model = load_bert_model(str(bert_dir))
            result = predict_text(bert_model, test_text, 'bert')
            print(f"Prediction: {result['label']}, Probability: {result['prob']:.4f}")
    except Exception as e:
        print(f"Could not load BERT model: {e}")

