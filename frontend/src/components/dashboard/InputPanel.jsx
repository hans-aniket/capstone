import { useState } from 'react';
import { Play, Loader2, Database } from 'lucide-react';

export default function InputPanel({ onAnalyze, isLoading, disabled }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !isLoading && !disabled) {
      onAnalyze(text.trim());
    }
  };

  const samplePayloads = [
    { label: 'Positive Target', text: "The integration was seamless and the performance improvements were immediately noticeable. Highly recommended architecture." },
    { label: 'Negative Target', text: "Significant latency regressions observed after the recent deployment. The system is unstable and documentation is outdated." },
    { label: 'Mixed Edge-case', text: "While the initial setup was straightforward, we're experiencing intermittent timeout issues during peak loads. Promising but needs stabilization." }
  ];

  return (
    <div className="panel p-5">
      <div className="flex flex-wrap items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
          <Database className="w-4 h-4 text-text-muted" />
          Inference Payload
        </h2>
        <div className="flex gap-2">
          {samplePayloads.map((payload, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setText(payload.text)}
              disabled={isLoading || disabled}
              className="px-2.5 py-1 text-[10px] uppercase tracking-wider font-medium rounded border border-border-subtle bg-bg-base text-text-muted hover:text-text-primary hover:border-border-strong transition-colors"
            >
              {payload.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Input unstructured text payload for comparative sentiment analysis..."
          className="w-full h-28 bg-bg-base border border-border-strong rounded-md p-3 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-indigo focus:ring-1 focus:ring-accent-indigo transition-colors resize-none font-mono"
          disabled={isLoading || disabled}
          spellCheck={false}
        />

        <div className="flex items-center justify-between">
          <div className="text-xs font-mono text-text-muted">
            {text.length} bytes
          </div>
          <button
            type="submit"
            disabled={!text.trim() || isLoading || disabled}
            className="btn-primary"
          >
            {isLoading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Processing...</>
            ) : (
              <><Play className="w-4 h-4 fill-current" /> Execute Pipeline</>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
