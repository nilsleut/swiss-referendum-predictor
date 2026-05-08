import { useEffect, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { api } from "../api/client";
import type { HistoryEntry, ModelRecord } from "../api/client";

export function AccuracyChart() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.history(500), api.models()])
      .then(([h, m]) => { setHistory(h); setModels(m); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="card"><p className="muted">Loading…</p></div>;

  // Scatter plot: only rows where actual is known
  const pairs = history
    .filter((r) => r.turnout_actual != null)
    .map((r) => ({ actual: r.turnout_actual!, predicted: r.turnout_predicted }));

  // Distribution bar chart for all predictions
  const bins: Record<string, number> = {};
  for (const r of history) {
    const bucket = `${Math.floor(r.turnout_predicted / 10) * 10}–${Math.floor(r.turnout_predicted / 10) * 10 + 10}`;
    bins[bucket] = (bins[bucket] ?? 0) + 1;
  }
  const distData = Object.entries(bins)
    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
    .map(([range, count]) => ({ range, count }));

  // Model metrics table
  const latestModel = models[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Model metrics */}
      {latestModel && (
        <div className="card">
          <h2 className="section-title">Model Performance — {latestModel.version}</h2>
          <div className="metrics-grid">
            {Object.entries(latestModel.metrics).map(([k, v]) => (
              <div key={k} className="metric-box">
                <span className="metric-label">{k.toUpperCase()}</span>
                <span className="metric-value">{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Predicted vs Actual scatter */}
      <div className="card">
        <h2 className="section-title">Predicted vs Actual Turnout</h2>
        {pairs.length === 0 ? (
          <p className="muted">
            No actuals recorded yet. Update <code>referendums.turnout_actual</code> in the DB to see this chart.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="actual"
                name="Actual"
                unit="%"
                domain={[0, 100]}
                label={{ value: "Actual (%)", position: "insideBottom", offset: -10 }}
              />
              <YAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                unit="%"
                domain={[0, 100]}
                label={{ value: "Predicted (%)", angle: -90, position: "insideLeft" }}
              />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
                stroke="#94a3b8"
                strokeDasharray="4 4"
                label="Perfect"
              />
              <Scatter data={pairs} fill="#2563eb" opacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Prediction distribution */}
      <div className="card">
        <h2 className="section-title">Prediction Distribution</h2>
        {distData.length === 0 ? (
          <p className="muted">No predictions yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={distData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="range" label={{ value: "Turnout range (%)", position: "insideBottom", offset: -10 }} />
              <YAxis label={{ value: "Count", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Bar dataKey="count" name="Predictions" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
