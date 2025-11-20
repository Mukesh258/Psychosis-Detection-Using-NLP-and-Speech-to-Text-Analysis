# Frontend - Psychosis Detection UI

React frontend for the psychosis detection tool with text and voice input capabilities.

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

**Note:** Make sure the backend server is running on `http://localhost:8000` before using the frontend.

## Features

- **Text Input**: Type or paste text for analysis
- **Voice Input**: Use Web Speech API for speech-to-text input
- **Real-time Prediction**: Get instant predictions with probability scores
- **Token Highlighting**: Visualize token-level importance with color gradients
- **Detailed Explanations**: View SHAP or attention-based feature contributions

## Project Structure

```
frontend/
├── package.json
├── public/
│   └── index.html
└── src/
    ├── App.jsx              # Main app component
    ├── index.jsx            # React entry point
    ├── components/
    │   ├── TextInput.jsx    # Text input component
    │   ├── VoiceInput.jsx   # Voice input (Web Speech API)
    │   ├── ResultCard.jsx   # Prediction results display
    │   ├── HighlightedText.jsx  # Token highlighting
    │   └── ExplanationPanel.jsx # Detailed explanations
    ├── styles/
    │   └── main.css         # Main stylesheet
    └── utils/
        └── api.js           # API client utilities
```

## Browser Compatibility

- **Web Speech API**: Supported in Chrome, Edge, Safari (macOS/iOS)
- **Voice input may not work in Firefox** - use text input as fallback

## Usage

1. Enter text in the text area, or click "Start Recording" to use voice input
2. Click "Analyze" to get predictions
3. Click "Explain" for detailed feature-level explanations
4. View token highlighting to see which words contribute to the prediction

## Ethical Disclaimer

This tool displays a visible ethical disclaimer banner indicating it is a research prototype and NOT a clinical diagnostic tool.

