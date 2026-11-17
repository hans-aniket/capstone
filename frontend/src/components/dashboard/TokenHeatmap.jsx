import React, { useState } from 'react';
import { Eye, ThumbsUp, ThumbsDown, Info, Cpu, Layers } from 'lucide-react';

export default function TokenHeatmap({ explanation, onModelChange, selectedModel }) {
  const [hoveredToken, setHoveredToken] = useState(null);

  if (!explanation || !explanation.tokens || explanation.tokens.length === 0) {
    return null;
  }

  const { model, tier, predicted_label, confidence, tokens, top_positive, top_negative } = explanation;
  const isPos = predicted_label.toLowerCase() === 'positive';

  // Compute background color based on attribution score and sentiment
  const getTokenStyle = (token) => {
    const intensity = Math.min(token.importance || Math.abs(token.score), 1.0);
    const alpha = Math.max(0.12, intensity * 0.7);

    if (token.sentiment === 'positive') {
      return {
        backgroundColor: `rgba(16, 185, 129, ${alpha})`,
        borderColor: `rgba(16, 185, 129, ${alpha + 0.2})`,
        color: '#10b981',
      };
    } else if (token.sentiment === 'negative') {
      return {
        backgroundColor: `rgba(244, 63, 94, ${alpha})`,
        borderColor: `rgba(244, 63, 94, ${alpha + 0.2})`,
        color: '#f43f5e',
      };
    }
    return {
      backgroundColor: 'transparent',
      borderColor: 'transparent',
      color: 'inherit',
    };
  };

  return (
    <div className="panel p-6 space-y-5 animate-fade-in border border-border-subtle bg-bg-panel/70">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border-subtle pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent-indigo/10 flex items-center justify-center border border-accent-indigo/20">
            <Eye className="w-4 h-4 text-accent-indigo" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              Explainability: Token Attribution Heatmap (XAI)
              <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-bg-base border border-border-subtle text-text-muted">
                {tier === 'transformers' ? 'Input x Gradient (GPU)' : 'TF-IDF Coefficient'}
              </span>
            </h3>
            <p className="text-xs text-text-muted mt-0.5">
              Visualizing how each token positively or negatively swayed {model}'s prediction.
            </p>
          </div>
        </div>

        {/* Model switcher */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-muted">Explaining:</span>
          <select
            value={selectedModel || model}
            onChange={(e) => onModelChange && onModelChange(e.target.value)}
            className="bg-bg-base border border-border-strong text-text-primary text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-accent-indigo"
          >
            <option value="DistilBERT">DistilBERT (Attention / Gradient)</option>
            <option value="Logistic Regression">Logistic Regression (TF-IDF)</option>
            <option value="SVM">Linear SVM (Margin Weight)</option>
            <option value="Naive Bayes">Naive Bayes (Log-Odds)</option>
          </select>
        </div>
      </div>

      {/* Heatmap Text Display */}
      <div className="p-4 rounded-lg bg-bg-base/80 border border-border-subtle relative min-h-[90px] leading-relaxed">
        <div className="flex flex-wrap gap-1.5 items-center text-sm">
          {tokens.map((token, idx) => {
            const style = getTokenStyle(token);
            const isHovered = hoveredToken === idx;

            return (
              <span
                key={idx}
                onMouseEnter={() => setHoveredToken(idx)}
                onMouseLeave={() => setHoveredToken(null)}
                style={style}
                className={`px-1.5 py-0.5 rounded border transition-all duration-150 cursor-pointer select-none font-medium text-xs sm:text-sm ${
                  isHovered ? 'ring-2 ring-accent-indigo shadow-md scale-105' : ''
                }`}
                title={`${token.word}: ${token.sentiment.toUpperCase()} (score: ${token.score})`}
              >
                {token.word}
              </span>
            );
          })}
        </div>

        {/* Hover inspector card */}
        {hoveredToken !== null && tokens[hoveredToken] && (
          <div className="mt-3 pt-2 border-t border-border-subtle flex items-center justify-between text-xs text-text-secondary animate-fade-in">
            <span className="font-mono text-text-primary font-semibold">
              "{tokens[hoveredToken].word}"
            </span>
            <div className="flex items-center gap-3 font-mono">
              <span>Sentiment: <strong className={tokens[hoveredToken].sentiment === 'positive' ? 'text-accent-emerald' : tokens[hoveredToken].sentiment === 'negative' ? 'text-accent-rose' : 'text-text-muted'}>{tokens[hoveredToken].sentiment.toUpperCase()}</strong></span>
              <span>Attribution Weight: <strong>{tokens[hoveredToken].score}</strong></span>
              <span>Relative Importance: <strong>{Math.round(tokens[hoveredToken].importance * 100)}%</strong></span>
            </div>
          </div>
        )}
      </div>

      {/* Key Drivers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
        <div className="p-3 rounded bg-emerald-500/5 border border-emerald-500/15 text-xs">
          <div className="flex items-center gap-1.5 text-accent-emerald font-semibold mb-2">
            <ThumbsUp className="w-3.5 h-3.5" />
            <span>Top Positive Influence Tokens</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {top_positive && top_positive.length > 0 ? (
              top_positive.map((w, i) => (
                <span key={i} className="px-2 py-0.5 rounded bg-emerald-500/10 text-accent-emerald border border-emerald-500/20 font-mono text-[11px]">
                  +{w}
                </span>
              ))
            ) : (
              <span className="text-text-muted italic">No strong positive tokens</span>
            )}
          </div>
        </div>

        <div className="p-3 rounded bg-rose-500/5 border border-rose-500/15 text-xs">
          <div className="flex items-center gap-1.5 text-accent-rose font-semibold mb-2">
            <ThumbsDown className="w-3.5 h-3.5" />
            <span>Top Negative Influence Tokens</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {top_negative && top_negative.length > 0 ? (
              top_negative.map((w, i) => (
                <span key={i} className="px-2 py-0.5 rounded bg-rose-500/10 text-accent-rose border border-rose-500/20 font-mono text-[11px]">
                  -{w}
                </span>
              ))
            ) : (
              <span className="text-text-muted italic">No strong negative tokens</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
