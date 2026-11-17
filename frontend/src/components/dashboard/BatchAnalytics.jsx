import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, TrendingUp, ThumbsUp, ThumbsDown, Download, RefreshCw, BarChart3, Database } from 'lucide-react';
import { uploadBatchCSV, getSampleBatch } from '../../services/api';

export default function BatchAnalytics() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('ALL');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const res = await uploadBatchCSV(file);
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getSampleBatch();
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExportCSV = () => {
    if (!data || !data.reviews) return;
    const headers = ["ID", "Product", "Review Text", "Sentiment", "Confidence (%)", "Primary Aspect"];
    const rows = data.reviews.map(r => [
      r.id,
      r.product ? `"${r.product.replace(/"/g, '""')}"` : "N/A",
      `"${r.text.replace(/"/g, '""')}"`,
      r.sentiment,
      r.confidence,
      r.primary_aspect
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `sentiment_batch_analysis_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredReviews = data?.reviews?.filter(r => {
    const matchesSearch = r.text.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          (r.product && r.product.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesSentiment = sentimentFilter === 'ALL' || r.sentiment.toUpperCase() === sentimentFilter;
    return matchesSearch && matchesSentiment;
  }) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Upload and Sample Panel */}
      <div className="panel p-6 border border-border-subtle bg-bg-panel/70 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-text-primary flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-accent-indigo" />
            Batch Reviews & E-Commerce Executive Analytics
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Process entire product catalogs or batches of Amazon customer reviews with GPU acceleration.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="cursor-pointer px-4 py-2 rounded-lg bg-bg-base hover:bg-bg-hover border border-border-strong text-text-primary text-xs font-medium flex items-center gap-2 transition-all">
            <Upload className="w-3.5 h-3.5 text-accent-indigo" />
            Upload CSV File
            <input type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" />
          </label>

          <button
            onClick={handleLoadSample}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white text-xs font-semibold flex items-center gap-2 transition-all shadow-sm"
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />}
            Load Sample Amazon Dataset (20 Reviews)
          </button>
        </div>
      </div>

      {error && (
        <div className="panel p-3 border-l-2 border-l-accent-rose bg-rose-500/5 text-xs text-accent-rose">
          {error}
        </div>
      )}

      {loading && !data && (
        <div className="panel p-12 text-center text-text-muted space-y-3">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto text-accent-indigo" />
          <p className="text-sm">Executing high-throughput batch inference and aspect decomposition on GPU...</p>
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {/* Executive KPI Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="panel p-4 border border-border-subtle bg-bg-panel/50">
              <span className="text-[10px] font-mono uppercase text-text-muted">Total Processed</span>
              <div className="text-2xl font-bold font-mono text-text-primary mt-1">
                {data.total_reviews} Reviews
              </div>
              <span className="text-xs text-text-secondary">Full batch analysis</span>
            </div>

            <div className="panel p-4 border border-border-subtle bg-bg-panel/50">
              <span className="text-[10px] font-mono uppercase text-text-muted">Sentiment Ratio</span>
              <div className="text-2xl font-bold font-mono text-accent-emerald mt-1">
                {data.positive_pct}% <span className="text-text-muted text-sm font-normal">/ {data.negative_pct}%</span>
              </div>
              <span className="text-xs text-text-secondary">{data.positive_count} Pos · {data.negative_count} Neg</span>
            </div>

            <div className="panel p-4 border border-border-subtle bg-bg-panel/50">
              <span className="text-[10px] font-mono uppercase text-text-muted">Net Promoter Score (NPS)</span>
              <div className={`text-2xl font-bold font-mono mt-1 ${
                data.nps_metrics.nps_score > 30 ? 'text-accent-emerald' : data.nps_metrics.nps_score > 0 ? 'text-accent-amber' : 'text-accent-rose'
              }`}>
                {data.nps_metrics.nps_score > 0 ? `+${data.nps_metrics.nps_score}` : data.nps_metrics.nps_score}
              </div>
              <span className="text-xs text-text-secondary">Tier: {data.nps_metrics.classification}</span>
            </div>

            <div className="panel p-4 border border-border-subtle bg-bg-panel/50">
              <span className="text-[10px] font-mono uppercase text-text-muted">Customer Stance Breakdown</span>
              <div className="flex items-center gap-2 mt-2">
                <span className="text-xs font-bold text-accent-emerald">{data.nps_metrics.promoters_pct}% Promoters</span>
                <span className="text-xs text-text-muted">·</span>
                <span className="text-xs font-bold text-accent-rose">{data.nps_metrics.detractors_pct}% Detractors</span>
              </div>
              <span className="text-xs text-text-muted">Passives: {data.nps_metrics.passives_pct}%</span>
            </div>
          </div>

          {/* Aspect Breakdown & Driver Keywords */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Aspect Satisfaction Breakdown */}
            <div className="panel p-5 border border-border-subtle bg-bg-panel/70 space-y-4">
              <h3 className="text-sm font-semibold text-text-primary">
                Aspect Satisfaction Breakdown Across Batch
              </h3>
              <div className="space-y-3">
                {data.aspect_breakdown && data.aspect_breakdown.map((asp, idx) => (
                  <div key={idx} className="space-y-1 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-text-secondary">{asp.aspect}</span>
                      <span className="font-mono font-bold text-text-primary">
                        {asp.satisfaction_rate}% Positive ({asp.total_mentions} mentions)
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-bg-base overflow-hidden flex">
                      <div
                        className="bg-accent-emerald h-full transition-all duration-300"
                        style={{ width: `${asp.satisfaction_rate}%` }}
                      />
                      <div
                        className="bg-accent-rose h-full transition-all duration-300"
                        style={{ width: `${100 - asp.satisfaction_rate}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Driver Keywords */}
            <div className="panel p-5 border border-border-subtle bg-bg-panel/70 space-y-4">
              <h3 className="text-sm font-semibold text-text-primary">
                Top Sentiment Drivers & Complaint Themes
              </h3>
              <div className="space-y-4">
                <div>
                  <span className="text-xs font-semibold text-accent-emerald flex items-center gap-1 mb-2">
                    <ThumbsUp className="w-3.5 h-3.5" /> Positive Product Highlights
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {data.top_positive_drivers.map((w, i) => (
                      <span key={i} className="px-2 py-0.5 rounded bg-emerald-500/10 text-accent-emerald border border-emerald-500/20 text-xs font-mono">
                        {w.word} ({w.count})
                      </span>
                    ))}
                  </div>
                </div>

                <div className="pt-2 border-t border-border-subtle">
                  <span className="text-xs font-semibold text-accent-rose flex items-center gap-1 mb-2">
                    <ThumbsDown className="w-3.5 h-3.5" /> Frequent Defect / Complaint Themes
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {data.top_complaint_themes.map((w, i) => (
                      <span key={i} className="px-2 py-0.5 rounded bg-rose-500/10 text-accent-rose border border-rose-500/20 text-xs font-mono">
                        {w.word} ({w.count})
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Processed Reviews Table */}
          <div className="panel p-6 border border-border-subtle bg-bg-panel/70 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Filter reviews by keyword..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-bg-base border border-border-strong text-text-primary text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-accent-indigo w-64"
                />
                <select
                  value={sentimentFilter}
                  onChange={(e) => setSentimentFilter(e.target.value)}
                  className="bg-bg-base border border-border-strong text-text-primary text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-accent-indigo"
                >
                  <option value="ALL">All Sentiments</option>
                  <option value="POSITIVE">Positive Only</option>
                  <option value="NEGATIVE">Negative Only</option>
                </select>
              </div>

              <button
                onClick={handleExportCSV}
                className="px-3 py-1.5 rounded-lg bg-bg-base hover:bg-bg-hover border border-border-strong text-text-primary text-xs font-medium flex items-center gap-1.5 transition-colors self-start sm:self-auto"
              >
                <Download className="w-3.5 h-3.5 text-accent-indigo" />
                Export Analyzed CSV
              </button>
            </div>

            <div className="overflow-x-auto max-h-[450px]">
              <table className="w-full text-xs text-left">
                <thead className="border-b border-border-subtle text-text-muted font-mono uppercase text-[10px] sticky top-0 bg-bg-panel">
                  <tr>
                    <th className="py-2.5 px-3">#</th>
                    <th className="py-2.5 px-3">Product</th>
                    <th className="py-2.5 px-3">Review</th>
                    <th className="py-2.5 px-3">Sentiment</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">Primary Aspect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle/50 font-mono">
                  {filteredReviews.map((r, idx) => (
                    <tr key={idx} className="hover:bg-bg-hover/40 transition-colors">
                      <td className="py-3 px-3 text-text-muted">{r.id}</td>
                      <td className="py-3 px-3 font-sans font-medium text-text-primary whitespace-nowrap">
                        {r.product || 'Amazon Item'}
                      </td>
                      <td className="py-3 px-3 font-sans text-text-secondary max-w-md line-clamp-2">
                        "{r.text}"
                      </td>
                      <td className="py-3 px-3">
                        <span className={`px-2 py-0.5 rounded font-bold ${
                          r.sentiment === 'Positive' ? 'bg-emerald-500/10 text-accent-emerald' : 'bg-rose-500/10 text-accent-rose'
                        }`}>
                          {r.sentiment}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-text-primary">{r.confidence}%</td>
                      <td className="py-3 px-3">
                        <span className="text-[11px] px-2 py-0.5 rounded bg-bg-base border border-border-subtle text-text-muted">
                          {r.primary_aspect}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
