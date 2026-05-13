import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';

export default function Visualizations({ predictions }) {
  const latencyData = [...predictions].sort((a, b) => a.latency_ms - b.latency_ms);
  
  const confData = [...predictions].map(p => ({
    name: p.model,
    value: p.confidence * 100,
    isPositive: p.label === 1
  }));

  const getTierColor = (tier) => {
    switch(tier) {
      case 'classical': return '#6366f1'; // indigo
      case 'deep_learning': return '#f59e0b'; // amber
      case 'transformers': return '#10b981'; // emerald
      default: return '#71717a';
    }
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-bg-panel border border-border-strong p-2.5 rounded shadow-lg text-xs font-sans">
          <p className="font-semibold text-text-primary mb-1">{payload[0].payload.model || payload[0].payload.name}</p>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: payload[0].fill }}></span>
            <span className="text-text-secondary font-mono">
              {payload[0].value.toFixed(2)} {payload[0].dataKey === 'latency_ms' ? 'ms' : '%'}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid lg:grid-cols-2 gap-4 animate-slide-down">
      {/* Latency Chart */}
      <div className="panel p-4 flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Latency Profiling</h4>
          <span className="text-[10px] text-text-muted font-mono">ms</span>
        </div>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={latencyData} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#27272a" />
              <XAxis type="number" stroke="#71717a" fontSize={10} tickFormatter={(val) => val.toFixed(0)} tickLine={false} axisLine={false} />
              <YAxis dataKey="model" type="category" stroke="#a1a1aa" fontSize={10} width={90} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: '#27272a' }} />
              <Bar dataKey="latency_ms" radius={[0, 2, 2, 0]} barSize={12}>
                {latencyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getTierColor(entry.tier)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confidence Chart */}
      <div className="panel p-4 flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Confidence Distribution</h4>
          <span className="text-[10px] text-text-muted font-mono">% prob</span>
        </div>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={confData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#27272a" />
              <XAxis dataKey="name" stroke="#a1a1aa" fontSize={10} interval={0} angle={-30} textAnchor="end" height={50} tickLine={false} axisLine={false} />
              <YAxis stroke="#71717a" fontSize={10} domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: '#27272a' }} />
              <Bar dataKey="value" radius={[2, 2, 0, 0]} barSize={16}>
                {confData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.isPositive ? '#10b981' : '#f43f5e'} opacity={0.9} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
