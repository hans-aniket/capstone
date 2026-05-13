import { Check, X } from 'lucide-react';
import clsx from 'clsx';

export default function ResultsTable({ predictions }) {
  const tierOrder = { classical: 1, deep_learning: 2, transformers: 3 };
  const sorted = [...predictions].sort((a, b) => 
    (tierOrder[a.tier] || 99) - (tierOrder[b.tier] || 99)
  );

  return (
    <div className="panel overflow-x-auto">
      <div className="px-4 py-3 border-b border-border-strong bg-bg-base/50">
        <h3 className="text-sm font-semibold text-text-primary">Inference Topology</h3>
      </div>
      <table className="w-full text-left text-sm whitespace-nowrap">
        <thead>
          <tr className="border-b border-border-subtle text-text-muted bg-bg-panel">
            <th className="py-2.5 px-4 font-medium text-[10px] uppercase tracking-wider">Model Architecture</th>
            <th className="py-2.5 px-4 font-medium text-[10px] uppercase tracking-wider">Classification</th>
            <th className="py-2.5 px-4 font-medium text-[10px] uppercase tracking-wider text-right">Probability</th>
            <th className="py-2.5 px-4 font-medium text-[10px] uppercase tracking-wider text-right">Latency</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border-subtle bg-bg-base">
          {sorted.map((pred, idx) => (
            <tr key={idx} className="group hover:bg-bg-hover transition-colors">
              <td className="py-2.5 px-4">
                <div className="text-sm font-semibold text-text-primary">
                  {pred.model}
                </div>
                <div className="text-[10px] text-text-muted uppercase tracking-widest font-medium mt-0.5">
                  {pred.tier.replace('_', ' ')}
                </div>
              </td>
              <td className="py-2.5 px-4">
                <div className={clsx(
                  "inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide",
                  pred.label === 1 
                    ? "text-accent-emerald bg-emerald-500/10" 
                    : "text-accent-rose bg-rose-500/10"
                )}>
                  {pred.label === 1 ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                  {pred.label_name}
                </div>
              </td>
              <td className="py-2.5 px-4 text-right">
                <div className="flex items-center justify-end gap-3">
                  <div className="w-16 h-1 bg-border-strong rounded-full overflow-hidden">
                    <div 
                      className={clsx("h-full rounded-full", pred.label === 1 ? "bg-accent-emerald" : "bg-accent-rose")}
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-text-secondary text-xs w-10">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </td>
              <td className="py-2.5 px-4 text-right">
                <span className={clsx(
                  "font-mono text-xs px-2 py-0.5 rounded border",
                  pred.latency_ms < 5 ? "text-text-primary border-border-subtle" :
                  pred.latency_ms < 30 ? "text-accent-amber border-amber-500/20 bg-amber-500/10" :
                  "text-accent-rose border-rose-500/20 bg-rose-500/10"
                )}>
                  {pred.latency_ms.toFixed(1)} ms
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
