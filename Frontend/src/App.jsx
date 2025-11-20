import React, { useState } from 'react';
import TextInput from './components/TextInput';
import VoiceInput from './components/VoiceInput';
import ResultCard from './components/ResultCard';
import HighlightedText from './components/HighlightedText';
import ExplanationPanel from './components/ExplanationPanel';
import './styles/main.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputText, setInputText] = useState('');

  const handleTextChange = (text) => {
    setInputText(text);
  };

  const handlePrediction = async (result) => {
    setPrediction(result);
    setError(null);
  };

  const handleExplanation = async (result) => {
    setExplanation(result);
  };

  const handleError = (err) => {
    setError(err);
    setPrediction(null);
    setExplanation(null);
  };

  const handleClear = () => {
    setPrediction(null);
    setExplanation(null);
    setError(null);
    setInputText('');
  };

  return (
    <div className="app">
      <div className="ethical-banner">
        <strong>⚠️ Ethical Disclaimer:</strong> This tool is a research prototype and NOT a clinical diagnostic tool.
      </div>
      
      <header className="app-header">
        <h1>Psychosis Detection Tool</h1>
        <p className="subtitle">Research Prototype for Text Analysis</p>
      </header>

      <main className="app-main">
        <div className="input-panel">
          <h2>Input Text</h2>
          <TextInput 
            value={inputText}
            onChange={handleTextChange}
            onPredict={handlePrediction}
            onExplain={handleExplanation}
            onError={handleError}
            loading={loading}
            setLoading={setLoading}
          />
          
          <div className="divider">OR</div>
          
          <VoiceInput
            onTranscript={handleTextChange}
            onPredict={handlePrediction}
            onError={handleError}
            loading={loading}
            setLoading={setLoading}
          />

          {error && (
            <div className="error-message">
              Error: {error}
            </div>
          )}

          {(prediction || explanation) && (
            <button className="clear-button" onClick={handleClear}>
              Clear Results
            </button>
          )}
        </div>

        <div className="output-panel">
          <h2>Analysis Results</h2>
          {loading && (
            <div className="loading">
              Analyzing text...
            </div>
          )}
          
          {prediction && (
            <>
              <ResultCard 
                label={prediction.label}
                prob={prediction.prob}
                probs={prediction.probs}
              />
              
              <HighlightedText
                text={inputText}
                tokens={prediction.tokens}
                tokenImportances={prediction.token_importances}
              />
            </>
          )}

          {explanation && (
            <ExplanationPanel explanation={explanation} />
          )}

          {!prediction && !explanation && !loading && (
            <div className="placeholder">
              Enter text above or use voice input to get started
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>This is a research tool. Not for clinical use.</p>
      </footer>
    </div>
  );
}

export default App;

