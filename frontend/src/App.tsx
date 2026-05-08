import { useState } from "react";
import { PredictionForm } from "./components/PredictionForm";
import { ResultCard } from "./components/ResultCard";
import { HistoryTable } from "./components/HistoryTable";
import { AccuracyChart } from "./components/AccuracyChart";
import type { PredictResponse } from "./api/client";

type Tab = "predict" | "history" | "analytics";

export function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [lastResult, setLastResult] = useState<PredictResponse | null>(null);

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div>
            <h1 className="header-title">Swiss Referendum Predictor</h1>
            <p className="header-sub">ML-powered voter turnout forecasting</p>
          </div>
          <div className="header-badge">CH</div>
        </div>
      </header>

      <nav className="tab-bar">
        {(["predict", "history", "analytics"] as Tab[]).map((t) => (
          <button
            key={t}
            className={`tab-btn${tab === t ? " tab-btn--active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      <main className="main">
        {tab === "predict" && (
          <div className="predict-layout">
            <PredictionForm
              onResult={(r) => {
                setLastResult(r);
              }}
            />
            {lastResult && <ResultCard result={lastResult} />}
          </div>
        )}
        {tab === "history" && <HistoryTable />}
        {tab === "analytics" && <AccuracyChart />}
      </main>
    </div>
  );
}
