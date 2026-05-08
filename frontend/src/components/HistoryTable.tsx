import { useEffect, useState, useCallback } from "react";
import { api } from "../api/client";
import type { HistoryEntry } from "../api/client";

type SortKey = keyof Pick<
  HistoryEntry,
  "prediction_time" | "turnout_predicted" | "turnout_actual"
>;

export function HistoryTable() {
  const [rows, setRows] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("prediction_time");
  const [sortAsc, setSortAsc] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setRows(await api.history(100));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) setSortAsc((a) => !a);
    else { setSortKey(key); setSortAsc(false); }
  }

  const sorted = [...rows].sort((a, b) => {
    const av = a[sortKey] ?? 0;
    const bv = b[sortKey] ?? 0;
    return sortAsc
      ? av < bv ? -1 : 1
      : av > bv ? -1 : 1;
  });

  function SortArrow({ col }: { col: SortKey }) {
    if (col !== sortKey) return <span style={{ opacity: 0.3 }}> ↕</span>;
    return <span>{sortAsc ? " ↑" : " ↓"}</span>;
  }

  if (loading) return <div className="card"><p className="muted">Loading…</p></div>;
  if (error) return <div className="card"><p className="error-text">{error}</p></div>;
  if (!rows.length) return (
    <div className="card">
      <h2 className="section-title">History</h2>
      <p className="muted">No predictions yet. Make one above!</p>
    </div>
  );

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 className="section-title" style={{ margin: 0 }}>History</h2>
        <button className="btn btn-secondary" onClick={() => void load()}>Refresh</button>
      </div>
      <div style={{ overflowX: "auto", marginTop: "1rem" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Title</th>
              <th>Category</th>
              <th
                className="sortable"
                onClick={() => toggleSort("turnout_predicted")}
              >
                Predicted<SortArrow col="turnout_predicted" />
              </th>
              <th
                className="sortable"
                onClick={() => toggleSort("turnout_actual")}
              >
                Actual<SortArrow col="turnout_actual" />
              </th>
              <th>Interval</th>
              <th>Model</th>
              <th
                className="sortable"
                onClick={() => toggleSort("prediction_time")}
              >
                Time<SortArrow col="prediction_time" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              return (
                <tr key={row.prediction_id}>
                  <td className="muted">#{row.prediction_id}</td>
                  <td>{row.title ?? "—"}</td>
                  <td>{row.topic_category ?? "—"}</td>
                  <td><strong>{row.turnout_predicted}%</strong></td>
                  <td>{row.turnout_actual != null ? `${row.turnout_actual}%` : <span className="muted">—</span>}</td>
                  <td className="muted" style={{ fontSize: "0.8rem" }}>
                    {row.confidence_low != null
                      ? `${row.confidence_low}% – ${row.confidence_high}%`
                      : "—"}
                  </td>
                  <td className="muted">{row.model_version}</td>
                  <td className="muted" style={{ fontSize: "0.8rem" }}>
                    {new Date(row.prediction_time).toLocaleString()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
