import React, { useState } from 'react';
import { predictText, explainText } from '../utils/api';

function TextInput({ value, onChange, onPredict, onExplain, onError, loading, setLoading }) {
  const [text, setText] = useState(value || '');

  React.useEffect(() => {
    setText(value || '');
  }, [value]);

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    onChange(newText);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) {
      onError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    onError(null);

    try {
      const result = await predictText(text);
      onPredict(result);
    } catch (err) {
      onError(err.message || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    if (!text.trim()) {
      onError('Please enter some text to explain');
      return;
    }

    setLoading(true);
    onError(null);

    try {
      const result = await explainText(text);
      onExplain(result);
    } catch (err) {
      onError(err.message || 'Failed to get explanation');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAnalyze();
    }
  };

  return (
    <div className="text-input-container">
      <textarea
        className="text-input"
        value={text}
        onChange={handleTextChange}
        onKeyDown={handleKeyPress}
        placeholder="Enter text to analyze here... (Ctrl/Cmd+Enter to analyze)"
        rows={8}
        disabled={loading}
      />
      <div className="button-group">
        <button 
          className="analyze-button"
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
        <button 
          className="explain-button"
          onClick={handleExplain}
          disabled={loading || !text.trim()}
        >
          Explain
        </button>
      </div>
    </div>
  );
}

export default TextInput;

