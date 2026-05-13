import { useState, useEffect } from 'react';
import Navbar from '../components/layout/Navbar';
import InputPanel from '../components/dashboard/InputPanel';
import KPIRow from '../components/dashboard/KPIRow';
import ResultsTable from '../components/dashboard/ResultsTable';
import Visualizations from '../components/dashboard/Visualizations';
import ExpandableJSON from '../components/dashboard/ExpandableJSON';
import { analyzeSentiment, checkHealth } from '../services/api';
import { AlertTriangle, Fingerprint } from 'lucide-react';

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState({ status: 'checking', loaded_models: [] });

  useEffect(() => {
    checkHealth().then(setHealth);
  }, []);

  const handleAnalyze = async (text) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeSentiment(text);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg-base text-text-primary flex flex-col">
      <Navbar health={health} />
      
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-6">
        <InputPanel 
          onAnalyze={handleAnalyze} 
          isLoading={loading} 
          disabled={health.status !== 'online'} 
        />
        
        {error && (
          <div className="panel p-3 border-l-2 border-l-accent-rose bg-rose-500/5 flex items-start gap-3 text-sm animate-fade-in">
            <AlertTriangle className="w-4 h-4 text-accent-rose shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-accent-rose">Pipeline Error</p>
              <p className="text-rose-200/80 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {!results && !loading && (
          <div className="panel p-12 flex flex-col items-center justify-center min-h-[300px] border-dashed animate-fade-in">
            <div className="w-10 h-10 rounded-full bg-bg-hover flex items-center justify-center mb-4">
              <Fingerprint className="w-4 h-4 text-text-muted" />
            </div>
            <h3 className="text-sm font-semibold text-text-primary">Awaiting Input Payload</h3>
            <p className="text-xs text-text-muted mt-1 max-w-sm text-center leading-relaxed">
              Submit unstructured text above to execute the comparative inference pipeline across all provisioned neural networks and classical classifiers.
            </p>
          </div>
        )}

        {results && (
          <div className="space-y-6">
            <KPIRow results={results} />
            <ResultsTable predictions={results.predictions} />
            <Visualizations predictions={results.predictions} />
            <ExpandableJSON data={results} />
          </div>
        )}
      </main>
    </div>
  );
}
