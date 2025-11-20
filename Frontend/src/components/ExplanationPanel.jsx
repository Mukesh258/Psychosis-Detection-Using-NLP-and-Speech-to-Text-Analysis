import React, { useMemo } from 'react';

function ExplanationPanel({ explanation }) {
  const sortedTokens = useMemo(() => {
    if (!explanation || !explanation.tokens || !explanation.importances) {
      return [];
    }

    const tokens = explanation.tokens || [];
    const importances = explanation.importances || [];

    // Combine and sort by absolute importance
    const combined = tokens.map((token, idx) => ({
      token,
      importance: importances[idx] || 0
    }));

    return combined.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
  }, [explanation]);

  const topPositive = sortedTokens.filter(item => item.importance > 0).slice(0, 10);
  const topNegative = sortedTokens.filter(item => item.importance < 0).slice(0, 10);

  return (
    <div className="explanation-panel">
      <h3>Detailed Explanation</h3>
      
      {explanation.summary && (
        <div className="explanation-summary">
          {explanation.summary}
        </div>
      )}

      <div className="explanation-content">
        <div className="explanation-section">
          <h4>Top Contributing Features (Psychotic-like)</h4>
          {topPositive.length > 0 ? (
            <ul className="feature-list positive">
              {topPositive.map((item, idx) => (
                <li key={idx}>
                  <span className="token-name">{item.token}</span>
                  <span className="token-score positive">
                    +{item.importance.toFixed(4)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-features">No positive contributions found</p>
          )}
        </div>

        <div className="explanation-section">
          <h4>Top Contributing Features (Normal)</h4>
          {topNegative.length > 0 ? (
            <ul className="feature-list negative">
              {topNegative.map((item, idx) => (
                <li key={idx}>
                  <span className="token-name">{item.token}</span>
                  <span className="token-score negative">
                    {item.importance.toFixed(4)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-features">No negative contributions found</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ExplanationPanel;

