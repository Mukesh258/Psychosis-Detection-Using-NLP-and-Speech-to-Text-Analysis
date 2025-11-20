"""
Explainability utilities using SHAP for TF-IDF models and integrated gradients for transformers.
"""
import numpy as np
import shap
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from scipy.special import expit


def explain_baseline_shap(model_data: dict, text: str, num_samples: int = 100):
    """
    Generate SHAP explanations for baseline models.
    
    Args:
        model_data: Dictionary containing model components
        text: Input text string
        num_samples: Number of samples for SHAP
        
    Returns:
        Dictionary with tokens and their importance scores
    """
    model = model_data['model']
    model_type = model_data['type']
    
    # Get vectorizer
    if model_type == 'pipeline':
        vectorizer = model.named_steps['tfidf']
        clf = model.named_steps['clf']
    else:  # xgboost
        vectorizer = model_data['vectorizer']
        clf = model_data['model']
    
    # Create SHAP explainer
    # Use KernelExplainer for any model
    def model_wrapper(X):
        """Wrapper function for SHAP."""
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X)
        else:
            # For models without predict_proba, use decision function
            prob = clf.decision_function(X)
            if prob.ndim == 1:
                # Binary classification - convert to probability
                prob_normal = expit(-prob)
                prob_psychotic = expit(prob)
                return np.column_stack([prob_normal, prob_psychotic])
            return prob
    
    # Get background data (sample from training if available, otherwise use zero vector)
    background = np.zeros((1, len(vectorizer.get_feature_names_out())))
    
    # Vectorize input
    X_input = vectorizer.transform([text])
    
    # Create explainer
    explainer = shap.KernelExplainer(model_wrapper, background)
    
    # Get SHAP values
    shap_values = explainer.shap_values(X_input.toarray(), nsamples=num_samples)
    
    # Handle multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get positive class
    
    # Get feature names and values
    feature_names = vectorizer.get_feature_names_out()
    token_scores = X_input.toarray()[0]
    
    # Create token-level explanations
    tokens = []
    importances = []
    
    # Get n-grams present in text
    for i, (feature_name, tfidf_val) in enumerate(zip(feature_names, token_scores)):
        if tfidf_val > 0:
            tokens.append(feature_name)
            importances.append(float(shap_values[0][i]))
    
    # Sort by importance
    sorted_indices = np.argsort(np.abs(importances))[::-1]
    tokens = [tokens[i] for i in sorted_indices[:30]]  # Top 30
    importances = [importances[i] for i in sorted_indices[:30]]
    
    return {
        'tokens': tokens,
        'importances': importances,
        'summary': f"SHAP explanation computed with {num_samples} samples"
    }


def explain_bert_integrated_gradients(model_data: dict, text: str, baseline_id: int = 0):
    """
    Generate integrated gradients explanations for DistilBERT.
    
    Args:
        model_data: Dictionary containing model components
        text: Input text string
        baseline_id: Token ID to use as baseline (0 = [PAD])
        
    Returns:
        Dictionary with tokens and their importance scores
    """
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Set model to eval mode
    model.eval()
    
    # Get embeddings
    embeddings = model.distilbert.embeddings(input_ids)
    
    # Create baseline (all zeros or pad tokens)
    baseline_ids = torch.full_like(input_ids, baseline_id)
    baseline_embeddings = model.distilbert.embeddings(baseline_ids)
    
    # Number of steps for integration
    steps = 50
    alphas = torch.linspace(0, 1, steps + 1)
    
    # Compute integrated gradients
    gradients = []
    
    for alpha in alphas:
        # Interpolated embedding
        interp_embeddings = baseline_embeddings + alpha * (embeddings - baseline_embeddings)
        
        # Create new input
        interp_embeddings.requires_grad_(True)
        
        # Forward pass through rest of model
        # We need to manually pass through distilbert layers
        hidden_state = interp_embeddings
        for layer in model.distilbert.transformer.layer:
            hidden_state = layer(hidden_state, attention_mask=attention_mask.unsqueeze(1).unsqueeze(2))
        
        hidden_state = hidden_state[:, 0]  # CLS token
        logits = model.pre_classifier(hidden_state)
        logits = F.relu(logits)
        logits = model.classifier(logits)
        
        # Get gradient w.r.t. embeddings
        score = logits[0, 1]  # Positive class score
        score.backward()
        
        grad = interp_embeddings.grad
        gradients.append(grad)
        
        # Clean up
        interp_embeddings.grad = None
    
    # Average gradients
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # Integrated gradients = (embeddings - baseline) * avg_gradients
    integrated_grads = (embeddings - baseline_embeddings) * avg_gradients
    
    # Sum over embedding dimension
    attributions = integrated_grads.sum(dim=-1).squeeze(0)
    
    # Convert to numpy
    attributions = attributions.detach().cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Filter out special tokens and get valid attributions
    valid_tokens = []
    valid_importances = []
    
    for i, (token, attn_mask, attr) in enumerate(zip(tokens, attention_mask[0], attributions)):
        if attn_mask.item() == 1 and token not in ['[CLS]', '[SEP]', '[PAD]']:
            valid_tokens.append(token.replace('##', ''))
            valid_importances.append(float(attr))
    
    # Normalize
    if valid_importances:
        max_imp = max(abs(imp) for imp in valid_importances)
        if max_imp > 0:
            valid_importances = [imp / max_imp for imp in valid_importances]
    
    return {
        'tokens': valid_tokens,
        'importances': valid_importances,
        'summary': 'Integrated gradients explanation computed'
    }


def explain_bert_attention(model_data: dict, text: str):
    """
    Generate attention-based explanations for DistilBERT (simpler alternative).
    
    Args:
        model_data: Dictionary containing model components
        text: Input text string
        
    Returns:
        Dictionary with tokens and their importance scores
    """
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    # Get attention
    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)
        attention = outputs.attentions[-1]  # Last layer
        # Average over all heads
        attention_avg = attention.mean(dim=1).squeeze(0)
        # Average attention received by each token
        token_attention = attention_avg.mean(dim=0).cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    # Filter valid tokens
    valid_tokens = []
    valid_importances = []
    
    for i, (token, attn) in enumerate(zip(tokens, token_attention)):
        if encoding['attention_mask'][0][i].item() == 1 and token not in ['[CLS]', '[SEP]', '[PAD]']:
            valid_tokens.append(token.replace('##', ''))
            valid_importances.append(float(attn))
    
    # Normalize
    if valid_importances:
        max_imp = max(valid_importances)
        if max_imp > 0:
            valid_importances = [imp / max_imp for imp in valid_importances]
    
    return {
        'tokens': valid_tokens,
        'importances': valid_importances,
        'summary': 'Attention-based explanation computed'
    }


if __name__ == "__main__":
    # Test explanation functions
    test_text = "I see patterns everywhere that others can't see. They're communicating through signs."
    
    print("Note: Explanation functions require loaded models to test.")
    print("Use explain_baseline_shap() for TF-IDF models")
    print("Use explain_bert_integrated_gradients() or explain_bert_attention() for BERT models")

