import React, { useState, useEffect } from 'react';
import { predictText } from '../utils/api';

function VoiceInput({ onTranscript, onPredict, onError, loading, setLoading }) {
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize Web Speech API
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setError('Web Speech API not supported in this browser');
      return;
    }

    const recognitionInstance = new SpeechRecognition();
    recognitionInstance.continuous = false;
    recognitionInstance.interimResults = false;
    recognitionInstance.lang = 'en-US';

    recognitionInstance.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognitionInstance.onresult = async (event) => {
      const currentTranscript = event.results[event.results.length - 1][0].transcript;
      setTranscript(currentTranscript);
      onTranscript(currentTranscript);
      setIsListening(false);

      // Automatically predict after transcript is received
      if (currentTranscript.trim()) {
        setLoading(true);
        try {
          const result = await predictText(currentTranscript);
          onPredict(result);
        } catch (err) {
          onError(err.message || 'Failed to get prediction');
        } finally {
          setLoading(false);
        }
      }
    };

    recognitionInstance.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setError(`Speech recognition error: ${event.error}`);
      setIsListening(false);
    };

    recognitionInstance.onend = () => {
      setIsListening(false);
    };

    setRecognition(recognitionInstance);

    return () => {
      if (recognitionInstance) {
        recognitionInstance.abort();
      }
    };
  }, [onTranscript, onPredict, onError, setLoading]);

  const handleStartListening = () => {
    if (recognition && !isListening) {
      try {
        recognition.start();
      } catch (err) {
        setError('Could not start speech recognition');
      }
    }
  };

  const handleStopListening = () => {
    if (recognition && isListening) {
      recognition.stop();
    }
  };

  return (
    <div className="voice-input-container">
      <div className="voice-controls">
        {!isListening ? (
          <button 
            className="voice-button start"
            onClick={handleStartListening}
            disabled={loading || !recognition}
            title="Start voice input"
          >
            🎤 Start Recording
          </button>
        ) : (
          <button 
            className="voice-button stop"
            onClick={handleStopListening}
            title="Stop recording"
          >
            ⏹ Stop Recording
          </button>
        )}
        
        {isListening && (
          <span className="listening-indicator">🔴 Listening...</span>
        )}
      </div>

      {transcript && (
        <div className="transcript-display">
          <strong>Transcript:</strong>
          <p>{transcript}</p>
        </div>
      )}

      {error && (
        <div className="voice-error">
          {error}
        </div>
      )}

      {!recognition && (
        <div className="voice-warning">
          Web Speech API is not supported in this browser. Please use Chrome or Edge.
        </div>
      )}
    </div>
  );
}

export default VoiceInput;

