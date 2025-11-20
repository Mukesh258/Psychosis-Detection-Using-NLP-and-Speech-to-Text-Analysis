"""
Prediction utilities combining model loading and explanation generation.
"""
from pathlib import Path

# Use explicit package imports so FastAPI/uvicorn can resolve modules reliably
from src.model_utils import load_baseline_model, load_bert_model, predict_text
from src.explain import explain_baseline_shap, explain_bert_attention
from src.preprocess import preprocess_text


class Predictor:
    """Main prediction class that handles model loading and prediction."""
    
    def __init__(self, model_type: str = 'baseline', model_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_type: 'baseline' or 'bert'
            model_path: Path to model file or directory
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model_data = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified model."""
        if self.model_path is None:
            script_dir = Path(__file__).parent
            models_dir = script_dir.parent / "models"
            
            if self.model_type == 'baseline':
                self.model_path = models_dir / "logistic_regression.pkl"
            elif self.model_type == 'bert':
                self.model_path = models_dir / "distilbert"
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
        
        if self.model_type == 'baseline':
            self.model_data = load_baseline_model(str(self.model_path))
        elif self.model_type == 'bert':
            self.model_data = load_bert_model(str(self.model_path))
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def predict(self, text: str, include_explanation: bool = True):
        """
        Predict label and probability for text.
        
        Args:
            text: Input text string
            include_explanation: Whether to include token-level explanations
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Get prediction
        result = predict_text(self.model_data, processed_text, self.model_type)
        
        # Add explanation if requested
        if include_explanation:
            if self.model_type == 'baseline':
                explanation = explain_baseline_shap(self.model_data, processed_text, num_samples=50)
            else:  # bert
                explanation = explain_bert_attention(self.model_data, processed_text)
            
            # Merge explanation into result
            result['explanation'] = explanation
        
        return result
    
    def explain(self, text: str):
        """
        Generate detailed explanation for text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with explanation results
        """
        processed_text = preprocess_text(text)
        
        if self.model_type == 'baseline':
            return explain_baseline_shap(self.model_data, processed_text, num_samples=100)
        else:  # bert
            return explain_bert_attention(self.model_data, processed_text)


if __name__ == "__main__":
    # Test predictor
    test_text = "I see patterns everywhere that others can't see."
    
    # Test baseline predictor
    try:
        print("Testing baseline predictor...")
        predictor = Predictor(model_type='baseline')
        result = predictor.predict(test_text)
        print(f"Label: {result['label']}")
        print(f"Probability: {result['prob']:.4f}")
        print(f"Tokens: {len(result['tokens'])}")
    except Exception as e:
        print(f"Baseline predictor test failed: {e}")
    
    # Test BERT predictor
    try:
        print("\nTesting BERT predictor...")
        predictor = Predictor(model_type='bert')
        result = predictor.predict(test_text)
        print(f"Label: {result['label']}")
        print(f"Probability: {result['prob']:.4f}")
        print(f"Tokens: {len(result['tokens'])}")
    except Exception as e:
        print(f"BERT predictor test failed: {e}")

