import { useState } from "react";
import { api, sampleFeatures, N_FEATURES } from "../api/client";
import type { PredictResponse } from "../api/client";

const CATEGORIES = [
  "Environment",
  "Economy",
  "Social",
  "Health",
  "Immigration",
  "Infrastructure",
  "Defense",
  "Other",
];

interface Props {
  onResult: (result: PredictResponse) => void;
}

export function PredictionForm({ onResult }: Props) {
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("");
  const [date, setDate] = useState("");
  const [featuresRaw, setFeaturesRaw] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function loadSample() {
    setTitle("Demo Referendum");
    setCategory("Environment");
    setFeaturesRaw(sampleFeatures().join(";"));
    setError(null);
  }

  function parseFeatures(): number[] | null {
    const sep = featuresRaw.includes(";") ? ";" : ",";
    const parts = featuresRaw.trim().split(sep).map((s) => parseFloat(s.trim()));
    if (parts.some(isNaN)) return null;
    return parts;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const features = parseFeatures();
    if (!features) {
      setError("Could not parse features — use semicolon-separated numbers.");
      return;
    }
    if (features.length !== N_FEATURES) {
      setError(`Need exactly ${N_FEATURES} features, got ${features.length}.`);
      return;
    }

    setLoading(true);
    try {
      const result = await api.predict({
        title: title.trim(),
        topic_category: category || undefined,
        date: date || undefined,
        features,
      });
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="card">
      <h2 className="section-title">New Prediction</h2>

      <div className="form-row">
        <label className="form-label">
          Title *
          <input
            className="form-input"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Energiegesetz"
            required
          />
        </label>

        <label className="form-label">
          Category
          <select
            className="form-input"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            <option value="">— none —</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c.toLowerCase()}>
                {c}
              </option>
            ))}
          </select>
        </label>

        <label className="form-label">
          Date
          <input
            type="date"
            className="form-input"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>
      </div>

      <label className="form-label" style={{ marginTop: "0.75rem" }}>
        Features ({N_FEATURES} semicolon-separated values)
        <textarea
          className="form-input"
          rows={4}
          value={featuresRaw}
          onChange={(e) => setFeaturesRaw(e.target.value)}
          placeholder="0;1;6190;228;0;0;…"
          style={{ fontFamily: "monospace", fontSize: "0.75rem" }}
        />
      </label>

      {error && <p className="error-text">{error}</p>}

      <div className="form-actions">
        <button type="button" className="btn btn-secondary" onClick={loadSample}>
          Load demo values
        </button>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? "Predicting…" : "Predict turnout"}
        </button>
      </div>
    </form>
  );
}
