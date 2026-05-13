import { Activity, Clock, Zap, Target } from 'lucide-react';
import clsx from 'clsx';

export default function KPIRow({ results }) {
  if (!results || !results.predictions?.length) return null;

  const preds = results.predictions;
  const posCount = preds.filter(p => p.label === 1).length;
  const isPositive = posCount > preds.length / 2;
  const agreementPct = (isPositive ? posCount : (preds.length - posCount)) / preds.length * 100;
  
  const sortedByLatency = [...preds].sort((a, b) => a.latency_ms - b.latency_ms);
  const fastestModel = sortedByLatency[0];
  
  const mostConfident = [...preds].sort((a, b) => 
    Math.abs(b.confidence - 0.5) - Math.abs(a.confidence - 0.5)
  )[0];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 animate-fade-in">
      {/* Consensus */}
      <div className="panel p-4 flex flex-col justify-between h-24 border-l-2 border-l-accent-indigo">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">System Consensus</span>
          <Activity className="w-3.5 h-3.5 text-text-muted" />
        </div>
        <div className="flex items-baseline justify-between mt-2">
          <span className={clsx("text-xl font-bold tracking-tight", isPositive ? "text-accent-emerald" : "text-accent-rose")}>
            {isPositive ? 'Positive' : 'Negative'}
          </span>
          <span className="text-xs text-text-secondary font-mono">{agreementPct.toFixed(0)}% agreement</span>
        </div>
      </div>

      {/* Latency */}
      <div className="panel p-4 flex flex-col justify-between h-24">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Total Latency</span>
          <Clock className="w-3.5 h-3.5 text-text-muted" />
        </div>
        <div className="flex items-baseline justify-between mt-2">
          <span className="text-xl font-bold text-text-primary tracking-tight">{results.total_latency_ms.toFixed(1)} <span className="text-sm font-medium text-text-muted">ms</span></span>
          <span className="text-[10px] text-text-secondary">End-to-end time</span>
        </div>
      </div>

      {/* Fastest Model */}
      <div className="panel p-4 flex flex-col justify-between h-24">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Peak Velocity</span>
          <Zap className="w-3.5 h-3.5 text-text-muted" />
        </div>
        <div className="flex flex-col mt-2">
          <span className="text-sm font-semibold text-text-primary truncate">{fastestModel.model}</span>
          <span className="text-xs text-text-secondary font-mono mt-0.5">{fastestModel.latency_ms.toFixed(2)} ms</span>
        </div>
      </div>

      {/* Most Confident */}
      <div className="panel p-4 flex flex-col justify-between h-24">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Highest Certainty</span>
          <Target className="w-3.5 h-3.5 text-text-muted" />
        </div>
        <div className="flex flex-col mt-2">
          <span className="text-sm font-semibold text-text-primary truncate">{mostConfident.model}</span>
          <span className="text-xs text-text-secondary font-mono mt-0.5">{(mostConfident.confidence * 100).toFixed(1)}% prob</span>
        </div>
      </div>
    </div>
  );
}
