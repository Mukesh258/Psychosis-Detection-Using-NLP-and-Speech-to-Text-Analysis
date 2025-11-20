"""
FastAPI backend server for psychosis detection.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import os

from src.predict import Predictor

app = FastAPI(title="Psychosis Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (default to baseline, can be changed to 'bert')
MODEL_TYPE = os.getenv("MODEL_TYPE", "baseline")
predictor = None

try:
    predictor = Predictor(model_type=MODEL_TYPE)
    print(f"Loaded {MODEL_TYPE} model successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Please train models first using train_baseline.py or train_transformer.py")


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    prob: float
    tokens: list
    token_importances: list
    probs: dict


class ExplanationResponse(BaseModel):
    tokens: list
    importances: list
    summary: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Psychosis Detection API",
        "model_type": MODEL_TYPE,
        "status": "ready" if predictor is not None else "model not loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Predict label and probability for input text.
    
    Input: { "text": "<string>" }
    Output: { "label": "psychotic-like"|"normal", "prob": float, "tokens": [...], "token_importances": [...] }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train models first.")
    
    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        result = predictor.predict(input_data.text, include_explanation=True)
        
        # Align tokens and importances from explanation if available
        tokens = result.get('tokens', [])
        token_importances = result.get('token_importances', [])
        
        # Use explanation tokens/importances if available and better
        if 'explanation' in result:
            exp_tokens = result['explanation'].get('tokens', [])
            exp_importances = result['explanation'].get('importances', [])
            
            if len(exp_tokens) > 0:
                tokens = exp_tokens
                token_importances = exp_importances
        
        return PredictionResponse(
            label=result['label'],
            prob=result['prob'],
            tokens=tokens[:50],  # Limit to 50 tokens
            token_importances=token_importances[:50],
            probs=result.get('probs', {result['label']: result['prob']})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(input_data: TextInput):
    """
    Generate detailed explanation for input text.
    
    Input: { "text": "<string>" }
    Output: SHAP values or integrated gradients style token importance
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train models first.")
    
    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        explanation = predictor.explain(input_data.text)
        return ExplanationResponse(
            tokens=explanation['tokens'][:50],
            importances=explanation['importances'][:50],
            summary=explanation.get('summary', 'Explanation computed')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/speech")
async def speech(input_data: dict):
    """
    Handle speech input (optional endpoint).
    
    Accepts either:
    - { "transcript": "<string>" } - returns /predict result
    - { "audio": "<base64>" } - for future Whisper integration (optional)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train models first.")
    
    if "transcript" in input_data:
        # If transcript is provided, use it directly
        text_input = TextInput(text=input_data["transcript"])
        return await predict(text_input)
    elif "audio" in input_data:
        # Audio processing would go here (optional Whisper integration)
        raise HTTPException(status_code=501, detail="Audio processing not yet implemented. Please provide transcript.")
    else:
        raise HTTPException(status_code=400, detail="Either 'transcript' or 'audio' field required")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

