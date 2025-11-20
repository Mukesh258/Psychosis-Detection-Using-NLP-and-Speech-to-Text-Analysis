# Psychosis Detection Tool

A full-stack NLP application for detecting psychotic-like behavior in text using machine learning models. This project consists of a Python FastAPI backend and a React frontend.

## ⚠️ Ethical Disclaimer

**This tool is a research prototype and NOT a clinical diagnostic tool.**

This application is for research purposes only and should not be used for clinical diagnosis or treatment decisions.

## Features

- **Text Analysis**: Analyze text input for psychotic-like patterns
- **Voice Input**: Speech-to-text using Web Speech API
- **Multiple Models**: Supports both baseline (TF-IDF + Logistic Regression/XGBoost) and transformer-based (DistilBERT) models
- **Explainability**: Token-level importance visualization using SHAP and attention mechanisms
- **Real-time Predictions**: Fast inference with probability scores
- **Modern UI**: Clean, responsive React interface

## Project Structure

```
psychosis_frontend_backend/
├── backend/              # Python FastAPI backend
│   ├── app.py           # FastAPI server
│   ├── requirements.txt # Python dependencies
│   ├── data/            # Dataset
│   ├── src/             # Source code
│   └── models/          # Saved models (created after training)
├── frontend/            # React frontend
│   ├── package.json     # Node dependencies
│   └── src/             # React source code
└── README.md            # This file
```

## Quick Start

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train baseline models:
   ```bash
   python src/train_baseline.py --data backend/data/synthetic_psychosis_data.csv
   ```

4. (Optional) Train DistilBERT model:
   ```bash
   python src/train_transformer.py --data backend/data/synthetic_psychosis_data.csv --epochs 3
   ```

5. Start the backend server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open `http://localhost:3000` in your browser

## Usage

1. **Text Input**: Type or paste text in the text area
2. **Voice Input**: Click "Start Recording" and speak (requires Web Speech API support)
3. **Analyze**: Click "Analyze" to get predictions
4. **Explain**: Click "Explain" for detailed feature-level explanations
5. **Visualize**: View token highlighting to see which words contribute to predictions

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

## Models

### Baseline Models
- **Logistic Regression** with TF-IDF features
- **XGBoost** with TF-IDF features
- Faster inference, suitable for production
- Uses SHAP for explainability

### Transformer Model
- **DistilBERT** fine-tuned on the dataset
- Better accuracy, requires more resources
- Uses attention mechanisms for explainability

## Dataset

The project uses a synthetic psychosis dataset located at:
`backend/data/synthetic_psychosis_data.csv`

The dataset contains text samples labeled as either "psychotic-like" or "normal".

## Development

### Backend Development

- FastAPI with automatic API documentation at `http://localhost:8000/docs`
- Models are saved in `backend/models/` after training
- Use environment variable `MODEL_TYPE=bert` to switch to DistilBERT

### Frontend Development

- React with functional components and hooks
- Uses Axios for API calls
- Web Speech API for voice input (Chrome/Edge recommended)

## Requirements

### Backend
- Python 3.8+
- See `backend/requirements.txt` for full list

### Frontend
- Node.js 16+
- npm or yarn
- See `frontend/package.json` for dependencies

## Browser Compatibility

- **Text Input**: All modern browsers
- **Voice Input**: Chrome, Edge, Safari (Web Speech API required)

## License

See LICENSE file for details.

## Contributing

This is a research project. For questions or issues, please refer to the repository.

## Acknowledgments

- Built with FastAPI, React, scikit-learn, Transformers, and SHAP
- Uses DistilBERT from Hugging Face Transformers

