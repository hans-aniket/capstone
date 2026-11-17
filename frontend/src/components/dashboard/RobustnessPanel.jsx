import React, { useState, useEffect } from 'react';
import { ShieldAlert, Zap, RefreshCw, AlertTriangle, Check, X, Sparkles, ChevronDown, ChevronRight } from 'lucide-react';
import { getRobustnessBenchmark, runStressTest } from '../../services/api';

export default function RobustnessPanel({ activeText }) {
  const [activeTab, setActiveTab] = useState('benchmark'); // 'benchmark' | 'interactive'
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [loadingBenchmark, setLoadingBenchmark] = useState(false);
  const [interactiveInput, setInteractiveInput] = useState(activeText || "The headphones have incredible battery life and sound great.");
  const [stressResults, setStressResults] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);
  const [expandedCase, setExpandedCase] = useState(null);

  useEffect(() => {
    loadBenchmark();
  }, []);

  useEffect(() => {
    if (activeText && activeText.trim()) {
      setInteractiveInput(activeText);
    }
  }, [activeText]);

  const loadBenchmark = async (forceRefresh = false) => {
    setLoadingBenchmark(true);
    try {
      const data = await getRobustnessBenchmark(forceRefresh);
      setBenchmarkData(data);
    } catch (err) {
      console.error("Failed to load benchmark:", err);
    } finally {
      setLoadingBenchmark(false);
    }
  };

  const handleRunInteractive = async () => {
    if (!interactiveInput.trim()) return;
    setStressLoading(true);
    try {
      const data = await runStressTest(interactiveInput);
      setStressResults(data);
    } catch (err) {
      console.error("Failed to run stress test:", err);
    } finally {
      setStressLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header Panel */}
      <div className="panel p-6 border border-border-subtle bg-bg-panel/70 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-accent-rose/10 flex items-center justify-center border border-accent-rose/20">
            <ShieldAlert className="w-5 h-5 text-accent-rose" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-text-primary">
              Adversarial, Sarcasm & Linguistic Robustness (CheckList Suite)
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Stress-testing invariance against negations, sarcastic irony, contrastive clauses, and noise perturbations.
            </p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center gap-2 p-1 bg-bg-base rounded-lg border border-border-strong text-xs font-medium">
          <button
            onClick={() => setActiveTab('benchmark')}
            className={`px-3 py-1.5 rounded transition-all ${
              activeTab === 'benchmark' ? 'bg-accent-indigo text-white shadow-sm font-semibold' : 'text-text-muted hover:text-text-primary'
            }`}
          >
            CheckList Benchmark
          </button>
          <button
            onClick={() => setActiveTab('interactive')}
            className={`px-3 py-1.5 rounded transition-all ${
              activeTab === 'interactive' ? 'bg-accent-indigo text-white shadow-sm font-semibold' : 'text-text-muted hover:text-text-primary'
            }`}
          >
            Interactive Stress Tester
          </button>
        </div>
      </div>

      {/* ── Mode 1: Benchmark Suite Leaderboard ── */}
      {activeTab === 'benchmark' && (
        <div className="space-y-6">
          {loadingBenchmark && !benchmarkData && (
            <div className="panel p-12 text-center text-text-muted space-y-3">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto text-accent-indigo" />
              <p className="text-sm">Running 20 curated challenge cases across all neural & classical models on GPU/CPU...</p>
            </div>
          )}

          {benchmarkData && (
            <>
              {/* Leaderboard Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {benchmarkData.leaderboard.map((item, idx) => {
                  const score = item.robustness_score;
                  const isTop = idx === 0;

                  return (
                    <div
                      key={idx}
                      className={`panel p-4 border relative overflow-hidden flex flex-col justify-between ${
                        isTop ? 'border-accent-indigo/40 bg-accent-indigo/5 ring-1 ring-accent-indigo/30' : 'border-border-subtle bg-bg-panel/50'
                      }`}
                    >
                      {isTop && (
                        <div className="absolute top-2 right-2 flex items-center gap-1 text-[10px] font-bold text-accent-indigo bg-accent-indigo/15 px-2 py-0.5 rounded">
                          <Sparkles className="w-3 h-3" /> Most Robust
                        </div>
                      )}
                      <div>
                        <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted">
                          {item.tier}
                        </span>
                        <h4 className="text-sm font-semibold text-text-primary mt-0.5">
                          {item.model}
                        </h4>
                      </div>

                      <div className="mt-4 pt-3 border-t border-border-subtle flex items-end justify-between">
                        <div>
                          <div className="text-2xl font-bold font-mono text-text-primary">
                            {score}%
                          </div>
                          <span className="text-[11px] text-text-muted">
                            Passed {item.passed} / {item.total} stress tests
                          </span>
                        </div>
                        <div className="w-12 h-1.5 rounded-full bg-bg-base overflow-hidden">
                          <div
                            className={`h-full rounded-full ${score > 70 ? 'bg-accent-emerald' : score > 50 ? 'bg-accent-amber' : 'bg-accent-rose'}`}
                            style={{ width: `${score}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Category Breakdown Table */}
              <div className="panel p-6 border border-border-subtle bg-bg-panel/70 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-text-primary">
                    Vulnerability Breakdown by Linguistic Category
                  </h3>
                  <button
                    onClick={() => loadBenchmark(true)}
                    className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" /> Re-run Suite
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="border-b border-border-subtle text-text-muted font-mono uppercase text-[10px]">
                      <tr>
                        <th className="py-2.5 px-3">Stress Vector</th>
                        {benchmarkData.leaderboard.map((m, i) => (
                          <th key={i} className="py-2.5 px-3 text-center">{m.model}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-subtle/50 font-mono">
                      {benchmarkData.category_breakdown.map((cat, i) => (
                        <tr key={i} className="hover:bg-bg-hover/50">
                          <td className="py-3 px-3 font-sans font-medium text-text-primary">
                            {cat.category}
                          </td>
                          {benchmarkData.leaderboard.map((m, j) => {
                            const stats = cat.models[m.model] || { accuracy_pct: 0, correct: 0, total: 0 };
                            const acc = stats.accuracy_pct;
                            return (
                              <td key={j} className="py-3 px-3 text-center">
                                <span className={`px-2 py-0.5 rounded font-bold ${
                                  acc >= 80 ? 'bg-emerald-500/10 text-accent-emerald' :
                                  acc >= 50 ? 'bg-amber-500/10 text-accent-amber' :
                                  'bg-rose-500/10 text-accent-rose'
                                }`}>
                                  {acc}% ({stats.correct}/{stats.total})
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Detailed Challenge Case Inspector */}
              <div className="panel p-6 border border-border-subtle bg-bg-panel/70 space-y-4">
                <h3 className="text-sm font-semibold text-text-primary">
                  Challenge Test Cases Explorer ({benchmarkData.cases.length} benchmark inputs)
                </h3>
                <div className="space-y-2">
                  {benchmarkData.cases.map((c, idx) => {
                    const isExpanded = expandedCase === idx;
                    return (
                      <div key={idx} className="border border-border-subtle rounded-lg bg-bg-base/60 overflow-hidden">
                        <div
                          onClick={() => setExpandedCase(isExpanded ? null : idx)}
                          className="p-3 flex items-center justify-between cursor-pointer hover:bg-bg-hover/50 transition-colors"
                        >
                          <div className="flex items-center gap-3">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                              c.expected === 'positive' ? 'bg-emerald-500/10 text-accent-emerald' : 'bg-rose-500/10 text-accent-rose'
                            }`}>
                              Expected: {c.expected}
                            </span>
                            <span className="text-xs text-text-primary font-medium line-clamp-1">
                              "{c.text}"
                            </span>
                          </div>

                          <div className="flex items-center gap-3">
                            <span className="text-[10px] text-text-muted px-2 py-0.5 rounded bg-bg-panel border border-border-subtle">
                              {c.category}
                            </span>
                            {isExpanded ? <ChevronDown className="w-4 h-4 text-text-muted" /> : <ChevronRight className="w-4 h-4 text-text-muted" />}
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="p-4 border-t border-border-subtle bg-bg-panel/40 space-y-3 text-xs animate-fade-in">
                            <p className="text-text-secondary">
                              <strong>Linguistic Trap:</strong> {c.description} ({c.difficulty} difficulty)
                            </p>
                            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2 pt-2">
                              {Object.entries(c.predictions).map(([mName, p], mIdx) => (
                                <div
                                  key={mIdx}
                                  className={`p-2 rounded border text-center font-mono ${
                                    p.correct ? 'bg-emerald-500/5 border-emerald-500/20 text-accent-emerald' : 'bg-rose-500/5 border-rose-500/20 text-accent-rose'
                                  }`}
                                >
                                  <div className="text-[10px] text-text-muted truncate">{mName}</div>
                                  <div className="font-bold uppercase text-xs mt-0.5 flex items-center justify-center gap-1">
                                    {p.correct ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                    {p.prediction || 'Error'}
                                  </div>
                                  <div className="text-[10px] text-text-muted">{Math.round((p.confidence || 0) * 100)}%</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Mode 2: Interactive Dynamic Perturbation Tester ── */}
      {activeTab === 'interactive' && (
        <div className="space-y-6">
          <div className="panel p-6 border border-border-subtle bg-bg-panel/70 space-y-4">
            <h3 className="text-sm font-semibold text-text-primary">
              Live Adversarial Perturbation Simulator
            </h3>
            <p className="text-xs text-text-muted leading-relaxed">
              Enter any review below. Our engine will dynamically inject character typos, sarcastic framing, and syntactic negations, then simultaneously evaluate all loaded models to test if predictions remain stable or flip.
            </p>

            <div className="space-y-3">
              <textarea
                value={interactiveInput}
                onChange={(e) => setInteractiveInput(e.target.value)}
                rows={3}
                className="w-full p-3 rounded-lg bg-bg-base border border-border-strong text-text-primary text-sm focus:outline-none focus:border-accent-indigo resize-none"
                placeholder="Type a product review..."
              />
              <button
                onClick={handleRunInteractive}
                disabled={stressLoading || !interactiveInput.trim()}
                className="px-4 py-2 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white font-medium text-xs flex items-center gap-2 transition-all disabled:opacity-50"
              >
                {stressLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
                Run Live Stress Test
              </button>
            </div>
          </div>

          {/* Stress Results Comparison Grid */}
          {stressResults && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in">
              {stressResults.results.map((v, idx) => (
                <div key={idx} className="panel p-5 border border-border-subtle bg-bg-panel/70 space-y-3 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-accent-indigo px-2 py-0.5 rounded bg-accent-indigo/10 border border-accent-indigo/20">
                        {v.perturbation_type}
                      </span>
                      <span className="text-[11px] text-text-muted">{v.description}</span>
                    </div>

                    <p className="text-xs text-text-primary mt-3 p-2.5 rounded bg-bg-base border border-border-subtle italic">
                      "{v.text}"
                    </p>
                  </div>

                  <div className="pt-2 border-t border-border-subtle space-y-1.5">
                    <span className="text-[10px] uppercase font-mono text-text-muted">Model Predictions:</span>
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      {Object.entries(v.predictions).map(([mName, p], i) => (
                        <div key={i} className="p-1.5 rounded bg-bg-base border border-border-subtle flex items-center justify-between">
                          <span className="truncate text-text-secondary text-[11px]">{mName}</span>
                          <span className={`text-[11px] font-bold ${
                            p.label?.toLowerCase() === 'positive' ? 'text-accent-emerald' : 'text-accent-rose'
                          }`}>
                            {p.label?.toUpperCase()} ({Math.round((p.confidence || 0) * 100)}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
