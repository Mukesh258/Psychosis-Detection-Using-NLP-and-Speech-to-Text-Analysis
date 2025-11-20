"""
Fine-tune DistilBERT for psychosis detection.
"""
import argparse
import pickle
import os
from pathlib import Path
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from transformers import (  # type: ignore
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from data_loader import load_dataset, get_data_splits
from preprocess import preprocess_batch


class PsychosisDataset(Dataset):
    """Dataset class for DistilBERT fine-tuning."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    
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
    
    # Encode labels (multi-class)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_labels = len(le.classes_)
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer and model: {model_name}...")
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure transformers and torch are installed: pip install transformers torch")
        raise
    
    # Create datasets
    train_dataset = PsychosisDataset(X_train, y_train_encoded, tokenizer, args.max_length)
    test_dataset = PsychosisDataset(X_test, y_test_encoded, tokenizer, args.max_length)
    
    # Training arguments
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)
    output_dir = models_dir / "distilbert"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",  # Using eval_strategy for better compatibility
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
    )
    
    # Create trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
    except TypeError as e:
        if "dispatch_batches" in str(e):
            print("\n" + "="*60)
            print("ERROR: Version compatibility issue with accelerate package")
            print("="*60)
            print("The transformers library requires accelerate >= 0.21.0")
            print("\nTo fix this, run:")
            print("  pip install accelerate>=0.21.0 --upgrade")
            print("\nOr reinstall all requirements:")
            print("  pip install -r requirements.txt --upgrade")
            print("="*60)
        raise
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Make predictions for detailed report
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, le.inverse_transform(y_pred)))
    
    # Save model and tokenizer
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save label encoder
    with open(models_dir / "distilbert_label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

