import { Server, Activity, AlertCircle } from 'lucide-react';

export default function Navbar({ health }) {
  const isOnline = health.status === 'online';
  const isChecking = health.status === 'checking';

  return (
    <header className="border-b border-border-strong bg-bg-base sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded bg-bg-panel border border-border-strong flex items-center justify-center shadow-sm">
            <Activity className="text-accent-indigo w-4 h-4" />
          </div>
          <h1 className="text-sm font-semibold text-text-primary tracking-tight">
            Comparative Sentiment Intelligence
          </h1>
        </div>
        
        <div className="flex items-center gap-2 text-xs font-medium px-2.5 py-1 rounded bg-bg-panel border border-border-subtle shadow-sm">
          {isOnline ? (
            <>
              <Server className="w-3.5 h-3.5 text-accent-emerald" />
              <span className="text-text-secondary hidden sm:inline">Engine Active</span>
              <span className="text-text-muted">({health.loaded_models?.length || 0} nodes)</span>
            </>
          ) : isChecking ? (
            <>
              <div className="w-2 h-2 rounded-full bg-accent-amber animate-pulse mx-1" />
              <span className="text-text-muted">Connecting...</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-3.5 h-3.5 text-accent-rose" />
              <span className="text-accent-rose">Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
