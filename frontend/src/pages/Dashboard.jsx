import { useState, useEffect } from 'react';
import Navbar from '../components/layout/Navbar';
import InputPanel from '../components/dashboard/InputPanel';
import KPIRow from '../components/dashboard/KPIRow';
import ResultsTable from '../components/dashboard/ResultsTable';
import Visualizations from '../components/dashboard/Visualizations';
import ExpandableJSON from '../components/dashboard/ExpandableJSON';
import TokenHeatmap from '../components/dashboard/TokenHeatmap';
import AspectBreakdown from '../components/dashboard/AspectBreakdown';
import RobustnessPanel from '../components/dashboard/RobustnessPanel';
import BatchAnalytics from '../components/dashboard/BatchAnalytics';
import { analyzeSentiment, checkHealth, getExplanation } from '../services/api';
import { AlertTriangle, Fingerprint, Activity, ShieldAlert, BarChart3 } from 'lucide-react';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('inference'); // 'inference' | 'robustness' | 'batch'
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState({ status: 'checking', loaded_models: [] });
  const [selectedExplainModel, setSelectedExplainModel] = useState('DistilBERT');
  const [currentText, setCurrentText] = useState('');

  useEffect(() => {
    checkHealth().then(setHealth);
  }, []);

  const handleAnalyze = async (text) => {
    setLoading(true);
    setError(null);
    setCurrentText(text);
    try {
      const data = await analyzeSentiment(text, selectedExplainModel);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleModelChangeExplain = async (newModel) => {
    setSelectedExplainModel(newModel);
    if (!currentText || !results) return;

    try {
      const explainData = await getExplanation(currentText, newModel);
      setResults(prev => ({
        ...prev,
        explanation: explainData
      }));
    } catch (err) {
      console.error("Failed to update explanation:", err);
    }
  };

  return (
    <div className="min-h-screen bg-bg-base text-text-primary flex flex-col">
      <Navbar health={health} />
      
      {/* Navigation Sub-header */}
      <div className="border-b border-border-subtle bg-bg-panel/40 backdrop-blur">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setActiveTab('inference')}
              className={`py-3 px-3.5 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
                activeTab === 'inference'
                  ? 'border-accent-indigo text-accent-indigo'
                  : 'border-transparent text-text-muted hover:text-text-primary'
              }`}
            >
              <Activity className="w-4 h-4" />
              Comparative Inference & XAI
            </button>

            <button
              onClick={() => setActiveTab('robustness')}
              className={`py-3 px-3.5 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
                activeTab === 'robustness'
                  ? 'border-accent-rose text-accent-rose'
                  : 'border-transparent text-text-muted hover:text-text-primary'
              }`}
            >
              <ShieldAlert className="w-4 h-4" />
              Robustness & Stress-Testing
            </button>

            <button
              onClick={() => setActiveTab('batch')}
              className={`py-3 px-3.5 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
                activeTab === 'batch'
                  ? 'border-accent-emerald text-accent-emerald'
                  : 'border-transparent text-text-muted hover:text-text-primary'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Batch Analytics & NPS
            </button>
          </div>
        </div>
      </div>

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-6">
        {activeTab === 'inference' && (
          <>
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
                  Submit unstructured product review text above to execute the comparative inference pipeline across all 7 neural networks and classical classifiers.
                </p>
              </div>
            )}

            {results && (
              <div className="space-y-6 animate-fade-in">
                <KPIRow results={results} />
                
                {/* Feature 1: Explainability Token Heatmap */}
                {results.explanation && (
                  <TokenHeatmap 
                    explanation={results.explanation} 
                    onModelChange={handleModelChangeExplain}
                    selectedModel={selectedExplainModel}
                  />
                )}

                {/* Feature 6: Aspect-Based Sentiment Analysis */}
                {results.aspect_analysis && (
                  <AspectBreakdown aspectAnalysis={results.aspect_analysis} />
                )}

                <ResultsTable predictions={results.predictions} />
                <Visualizations predictions={results.predictions} />
                <ExpandableJSON data={results} />
              </div>
            )}
          </>
        )}

        {/* Feature 2: Robustness Panel */}
        {activeTab === 'robustness' && (
          <RobustnessPanel activeText={currentText} />
        )}

        {/* Feature 5: Batch Analytics Panel */}
        {activeTab === 'batch' && (
          <BatchAnalytics />
        )}
      </main>
    </div>
  );
}
