import React, { useMemo } from 'react';

function HighlightedText({ text, tokens, tokenImportances }) {
  // Create a map of token to importance
  const tokenImportanceMap = useMemo(() => {
    if (!tokens || !tokenImportances || tokens.length !== tokenImportances.length) {
      return new Map();
    }

    const map = new Map();
    tokens.forEach((token, idx) => {
      const importance = tokenImportances[idx] || 0;
      // Token might be an n-gram, split it for matching
      const tokenParts = token.toLowerCase().split(/\s+/);
      tokenParts.forEach(part => {
        // Store max importance for each word part
        if (!map.has(part) || Math.abs(map.get(part)) < Math.abs(importance)) {
          map.set(part, importance);
        }
      });
    });
    return map;
  }, [tokens, tokenImportances]);

  // Tokenize and highlight text
  const highlightedText = useMemo(() => {
    if (!text || tokenImportanceMap.size === 0) {
      return [{ text, importance: 0 }];
    }

    // Simple word tokenization
    const words = text.split(/(\s+|[.,!?;:])/);
    return words.map(word => {
      const cleanWord = word.toLowerCase().replace(/[^\w]/g, '');
      const importance = tokenImportanceMap.get(cleanWord) || 0;
      return { text: word, importance };
    });
  }, [text, tokenImportanceMap]);

  const getColor = (importance) => {
    if (importance === 0) {
      return '#6b7280'; // Gray for neutral
    }

    // Normalize importance to 0-1 range for color gradient
    const normalized = Math.abs(importance);
    const clamped = Math.min(normalized, 1);

    if (importance > 0) {
      // Positive importance (psychotic-like) - red gradient
      const intensity = Math.floor(clamped * 255);
      return `rgb(239, ${68 - intensity}, ${68 - intensity})`;
    } else {
      // Negative importance (normal) - green gradient
      const intensity = Math.floor(clamped * 255);
      return `rgb(${16 + intensity}, 185, ${129 - intensity})`;
    }
  };

  const getBackgroundColor = (importance) => {
    if (importance === 0) {
      return 'transparent';
    }

    const normalized = Math.abs(importance);
    const clamped = Math.min(normalized, 1);
    const opacity = clamped * 0.3;

    if (importance > 0) {
      return `rgba(239, 68, 68, ${opacity})`;
    } else {
      return `rgba(16, 185, 129, ${opacity})`;
    }
  };

  return (
    <div className="highlighted-text-container">
      <h3>Token Importance Visualization</h3>
      <div className="highlighted-text">
        {highlightedText.map((item, idx) => (
          <span
            key={idx}
            className="highlighted-token"
            style={{
              color: getColor(item.importance),
              backgroundColor: getBackgroundColor(item.importance),
              fontWeight: Math.abs(item.importance) > 0.5 ? 'bold' : 'normal'
            }}
            title={item.importance !== 0 ? `Importance: ${item.importance.toFixed(4)}` : ''}
          >
            {item.text}
          </span>
        ))}
      </div>
      <div className="color-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: 'rgba(239, 68, 68, 0.3)' }}></span>
          <span>Higher psychotic-like probability</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: 'rgba(16, 185, 129, 0.3)' }}></span>
          <span>Higher normal probability</span>
        </div>
      </div>
    </div>
  );
}

export default HighlightedText;

