import React from 'react';

function ResultCard({ label, prob, probs }) {
  const topLabel = label || 'Unknown';
  const percentage = (prob * 100).toFixed(2);

  // Treat anything that is not clearly "Normal" as higher-risk for styling
  const isNormal = topLabel.toLowerCase() === 'normal';

  const getMessage = () => {
    const labelLower = topLabel.toLowerCase();

    if (prob >= 0.8) {
      return `High probability of ${labelLower} patterns detected.`;
    } else if (prob >= 0.6) {
      return `Moderate probability of ${labelLower} patterns detected.`;
    }
    return `Low probability of ${labelLower} patterns detected.`;
  };

  return (
    <div className={`result-card ${isNormal ? 'normal' : 'psychotic'}`}>
      <div className="result-header">
        <h3>Prediction</h3>
        <span className={`label-badge ${isNormal ? 'normal' : 'psychotic'}`}>
          {topLabel.toUpperCase()}
        </span>
      </div>

      <div className="probability-display">
        <div className="probability-bar-container">
          <div 
            className="probability-bar"
            style={{
              width: `${percentage}%`,
              backgroundColor: isNormal ? '#10b981' : '#ef4444'
            }}
          />
        </div>
        <div className="probability-value">
          {percentage}%
        </div>
      </div>

      <div className="result-message">
        {getMessage()}
      </div>

      {probs && (
        <div className="probability-breakdown">
          {Object.entries(probs).map(([cls, p]) => (
            <div className="prob-item" key={cls}>
              <span>{cls}:</span>
              <span>{(p * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="ethical-note">
        ⚠️ This is a research prototype and NOT a clinical diagnostic tool.
      </div>
    </div>
  );
}

export default ResultCard;

