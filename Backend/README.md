# Backend - Psychosis Detection API

FastAPI backend server for psychosis detection using NLP models.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

The first time you run the code, NLTK will download required data automatically. If needed manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

## Training Models

### Train Baseline Models (Logistic Regression + XGBoost)

Train TF-IDF based baseline models:

From the `backend` directory:
```bash
python src/train_baseline.py --data data/synthetic_psychosis_data.csv
```

Or from the project root:
```bash
cd backend
python src/train_baseline.py --data backend/data/synthetic_psychosis_data.csv
```

This will create:
- `models/logistic_regression.pkl`
- `models/xgboost.pkl`

### Train DistilBERT Model

Fine-tune DistilBERT for sequence classification:

From the `backend` directory:
```bash
python src/train_transformer.py --data data/synthetic_psychosis_data.csv --epochs 3 --batch_size 8
```

Or from the project root:
```bash
cd backend
python src/train_transformer.py --data backend/data/synthetic_psychosis_data.csv --epochs 3 --batch_size 8
```

Options:
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 8)
- `--max_length`: Maximum sequence length (default: 256)
- `--learning_rate`: Learning rate (default: 2e-5)

This will create:
- `models/distilbert/` (model + tokenizer)
- `models/distilbert_label_encoder.pkl`

## Running the Server

### Start the Backend Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Change Model Type

By default, the server uses the baseline model. To use DistilBERT:

```bash
export MODEL_TYPE=bert
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Or on Windows:
```cmd
set MODEL_TYPE=bert
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /predict

Predict label and probability for input text.

**Request:**
```json
{
  "text": "I see patterns everywhere that others can't see."
}
```

**Response:**
```json
{
  "label": "psychotic-like",
  "prob": 0.85,
  "tokens": ["patterns", "see", "others"],
  "token_importances": [0.45, 0.32, 0.23],
  "probs": {
    "normal": 0.15,
    "psychotic-like": 0.85
  }
}
```

### POST /explain

Generate detailed explanation for input text.

**Request:**
```json
{
  "text": "I see patterns everywhere that others can't see."
}
```

**Response:**
```json
{
  "tokens": ["patterns", "see", "others"],
  "importances": [0.45, 0.32, 0.23],
  "summary": "SHAP explanation computed with 100 samples"
}
```

### POST /speech

Handle speech input (optional endpoint).

**Request:**
```json
{
  "transcript": "I see patterns everywhere that others can't see."
}
```

**Response:** Same as `/predict`

## Project Structure

```
backend/
├── app.py                    # FastAPI server
├── requirements.txt          # Python dependencies
├── data/
│   └── synthetic_psychosis_data.csv
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocess.py        # Text preprocessing
│   ├── features.py          # Feature extraction
│   ├── train_baseline.py    # Train baseline models
│   ├── train_transformer.py # Fine-tune DistilBERT
│   ├── model_utils.py       # Model loading utilities
│   ├── explain.py           # Explainability functions
│   └── predict.py           # Prediction utilities
├── models/                  # Saved models (created after training)
└── notebooks/
    └── explore.ipynb        # Data exploration notebook
```

## Notes

- Make sure to train at least one model before starting the server
- The baseline models are faster but less accurate than DistilBERT
- DistilBERT requires more computational resources but provides better predictions
- Token-level explanations use SHAP for baseline models and attention for BERT models




Run commands

1. Train models:
   cd backend
   python src/train_baseline.py --data backend/data/synthetic_psychosis_data.csv
   python src/train_transformer.py --data backend/data/synthetic_psychosis_data.csv --epochs 3

2. Start backend:
   uvicorn app:app --reload --host 0.0.0.0 --port 8000

3. Start frontend:
   cd frontend
   npm install
   npm start