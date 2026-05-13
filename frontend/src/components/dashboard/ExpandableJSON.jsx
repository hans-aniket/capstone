import { Copy, Code, Download } from 'lucide-react';
import { useState } from 'react';

export default function ExpandableJSON({ data }) {
  const [isOpen, setIsOpen] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'inference_results.json';
    a.click();
  };

  return (
    <div className="panel animate-slide-down">
      <div className="flex items-center justify-between p-3 bg-bg-base border-b border-border-strong cursor-pointer" onClick={() => setIsOpen(!isOpen)}>
        <div className="flex items-center gap-2">
          <Code className="w-3.5 h-3.5 text-text-muted" />
          <span className="text-xs font-semibold text-text-primary uppercase tracking-wider">API Telemetry</span>
        </div>
        <button className="text-xs text-text-muted hover:text-text-primary transition-colors">
          {isOpen ? 'Collapse' : 'Expand'}
        </button>
      </div>

      {isOpen && (
        <div className="relative bg-[#09090b] p-4 max-h-80 overflow-auto border-t border-border-strong">
          <div className="absolute top-4 right-4 flex gap-2">
            <button onClick={handleExport} className="p-1.5 rounded bg-bg-panel border border-border-strong text-text-muted hover:text-text-primary transition-colors" title="Download JSON">
              <Download className="w-3.5 h-3.5" />
            </button>
            <button onClick={handleCopy} className="p-1.5 rounded bg-bg-panel border border-border-strong text-text-muted hover:text-text-primary transition-colors" title="Copy to clipboard">
              <Copy className="w-3.5 h-3.5" />
            </button>
          </div>
          <pre className="text-accent-indigo font-mono text-[10px] leading-relaxed">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
