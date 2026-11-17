import React from 'react';
import { Tag, BatteryCharging, ShieldCheck, Zap, DollarSign, Palette, Headphones, HelpCircle, CheckCircle2, XCircle } from 'lucide-react';

const ASPECT_ICONS = {
  'Battery & Charging': BatteryCharging,
  'Build Quality & Durability': ShieldCheck,
  'Performance & Speed': Zap,
  'Price & Value': DollarSign,
  'Design & Ergonomics': Palette,
  'Sound & Display / Media': Headphones,
  'Customer Support & Delivery': HelpCircle,
};

export default function AspectBreakdown({ aspectAnalysis }) {
  if (!aspectAnalysis || !aspectAnalysis.aspects || aspectAnalysis.aspects.length === 0) {
    return null;
  }

  const { aspects_found, dominant_tone, aspects } = aspectAnalysis;

  return (
    <div className="panel p-6 space-y-5 animate-fade-in border border-border-subtle bg-bg-panel/70">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border-subtle pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent-amber/10 flex items-center justify-center border border-accent-amber/20">
            <Tag className="w-4 h-4 text-accent-amber" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              Aspect-Based Sentiment Analysis (ABSA)
              <span className="text-[11px] font-normal px-2 py-0.5 rounded bg-bg-base border border-border-subtle text-text-muted">
                {aspects_found} {aspects_found === 1 ? 'aspect' : 'aspects'} detected
              </span>
            </h3>
            <p className="text-xs text-text-muted mt-0.5">
              Fine-grained multi-attribute sentiment decomposition from user review text.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-muted">Dominant Stance:</span>
          <span className={`px-2.5 py-1 rounded font-semibold border ${
            dominant_tone.includes('Positive') ? 'bg-emerald-500/10 text-accent-emerald border-emerald-500/20' :
            dominant_tone.includes('Negative') ? 'bg-rose-500/10 text-accent-rose border-rose-500/20' :
            'bg-amber-500/10 text-accent-amber border-amber-500/20'
          }`}>
            {dominant_tone}
          </span>
        </div>
      </div>

      {/* Grid of Aspect Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
        {aspects.map((item, idx) => {
          const Icon = ASPECT_ICONS[item.aspect] || Tag;
          const isPos = item.sentiment === 'Positive';
          const isNeg = item.sentiment === 'Negative';

          return (
            <div
              key={idx}
              className={`p-4 rounded-lg border bg-bg-base/70 transition-all duration-200 hover:border-border-strong flex flex-col justify-between space-y-3 ${
                isPos ? 'border-emerald-500/20' : isNeg ? 'border-rose-500/20' : 'border-border-subtle'
              }`}
            >
              <div>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div className={`w-7 h-7 rounded flex items-center justify-center ${
                      isPos ? 'bg-emerald-500/10 text-accent-emerald' : isNeg ? 'bg-rose-500/10 text-accent-rose' : 'bg-bg-hover text-text-muted'
                    }`}>
                      <Icon className="w-3.5 h-3.5" />
                    </div>
                    <span className="text-xs font-semibold text-text-primary">
                      {item.aspect}
                    </span>
                  </div>

                  <span className={`text-[11px] font-bold px-2 py-0.5 rounded flex items-center gap-1 ${
                    isPos ? 'bg-emerald-500/15 text-accent-emerald' : isNeg ? 'bg-rose-500/15 text-accent-rose' : 'bg-bg-hover text-text-muted'
                  }`}>
                    {isPos ? <CheckCircle2 className="w-3 h-3" /> : isNeg ? <XCircle className="w-3 h-3" /> : null}
                    {item.sentiment} ({Math.round(item.confidence * 100)}%)
                  </span>
                </div>

                {/* Extracted snippet */}
                <p className="text-xs text-text-secondary mt-3 italic line-clamp-2 bg-bg-panel/40 p-2 rounded border border-border-subtle/50">
                  "{item.snippet}"
                </p>
              </div>

              {/* Matched keywords */}
              <div className="flex items-center gap-1.5 pt-1">
                <span className="text-[10px] text-text-muted uppercase tracking-wider font-mono">Matched:</span>
                {item.keywords_matched.map((kw, i) => (
                  <span key={i} className="text-[10px] px-1.5 py-0.2 bg-bg-hover text-text-secondary rounded font-mono">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
