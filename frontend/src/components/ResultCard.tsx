import type { PredictResponse } from "../api/client";

interface Props {
  result: PredictResponse;
}

export function ResultCard({ result }: Props) {
  const turnout = result.predicted_turnout;
  const [lo, hi] = result.confidence_interval;

  // Colour the gauge needle: green >50%, orange 40-50%, red <40%
  const colour = turnout >= 50 ? "#22c55e" : turnout >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div className="card result-card">
      <h2 className="section-title">Prediction Result</h2>

      <div className="result-gauge">
        <svg viewBox="0 0 120 70" width="200" height="120">
          {/* Background arc */}
          <path
            d="M10 65 A55 55 0 0 1 110 65"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="10"
            strokeLinecap="round"
          />
          {/* Filled arc proportional to turnout */}
          <path
            d="M10 65 A55 55 0 0 1 110 65"
            fill="none"
            stroke={colour}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${(turnout / 100) * 172.8} 172.8`}
          />
          <text x="60" y="62" textAnchor="middle" fontSize="18" fontWeight="700" fill={colour}>
            {turnout}%
          </text>
        </svg>
      </div>

      <dl className="result-grid">
        <dt>Predicted turnout</dt>
        <dd style={{ color: colour, fontWeight: 700 }}>{turnout}%</dd>

        <dt>95% interval</dt>
        <dd>{lo}% – {hi}%</dd>

        <dt>Model version</dt>
        <dd>{result.model_version}</dd>

        <dt>Prediction ID</dt>
        <dd>#{result.prediction_id}</dd>

        <dt>Time</dt>
        <dd>{new Date(result.prediction_time).toLocaleString()}</dd>
      </dl>
    </div>
  );
}
