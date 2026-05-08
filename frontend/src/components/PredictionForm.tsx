import { useState } from "react";
import {
  api,
  CANTONS, THEMES, INSTITUTIONS, VOTE_TRIGGERS, VOTE_TRIGGER_ACTORS,
  VOTE_OBJECTS, AUTHORS, ACTIONS, LEGAL_NORM_HIERARCHIES, DEGREES_OF_REVISION,
  DECISION_QUORUMS, LEGAL_ACT_TYPES, SPECIAL_TOPICS,
} from "../api/client";
import type { PredictResponse, NamedFeatureInput } from "../api/client";

interface Props {
  onResult: (result: PredictResponse) => void;
}

function Select({
  label, value, onChange, options, placeholder = "— not specified —",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder?: string;
}) {
  return (
    <label className="form-label">
      {label}
      <select
        className="form-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">{placeholder}</option>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </label>
  );
}

export function PredictionForm({ onResult }: Props) {
  const [title, setTitle]       = useState("");
  const [date, setDate]         = useState("");
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Named feature state — all optional
  const [level, setLevel]                 = useState<"national" | "cantonal" | "">("");
  const [canton, setCanton]               = useState("");
  const [theme1, setTheme1]               = useState("");
  const [theme2, setTheme2]               = useState("");
  const [theme3, setTheme3]               = useState("");
  const [institution, setInstitution]     = useState("");
  const [counterProposal, setCounterProposal] = useState("");
  const [action, setAction]               = useState("");
  const [decisionQuorum, setDecisionQuorum] = useState("");
  const [legalNormHierarchy, setLegalNormHierarchy] = useState("");
  const [degreeOfRevision, setDegreeOfRevision] = useState("");
  const [legalActType, setLegalActType]   = useState("");
  const [voteTrigger, setVoteTrigger]     = useState("");
  const [voteTriggerActor, setVoteTriggerActor] = useState("");
  const [voteObject, setVoteObject]       = useState("");
  const [author, setAuthor]               = useState("");
  const [specialTopics, setSpecialTopics] = useState("");

  function buildNamedFeatures(): NamedFeatureInput {
    const f: NamedFeatureInput = {};
    if (level)              f.level              = level;
    if (canton)             f.canton             = canton;
    if (theme1)             f.theme1             = theme1;
    if (theme2)             f.theme2             = theme2;
    if (theme3)             f.theme3             = theme3;
    if (institution)        f.institution        = institution;
    if (counterProposal)    f.counter_proposal   = counterProposal;
    if (action)             f.action             = action;
    if (decisionQuorum)     f.decision_quorum    = decisionQuorum;
    if (legalNormHierarchy) f.legal_norm_hierarchy = legalNormHierarchy;
    if (degreeOfRevision)   f.degree_of_revision = degreeOfRevision;
    if (legalActType)       f.legal_act_type     = legalActType;
    if (voteTrigger)        f.vote_trigger       = voteTrigger;
    if (voteTriggerActor)   f.vote_trigger_actor = voteTriggerActor;
    if (voteObject)         f.vote_object        = voteObject;
    if (author)             f.author             = author;
    if (specialTopics)      f.special_topics     = specialTopics;
    return f;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await api.predict({
        title: title.trim(),
        date: date || undefined,
        named_features: buildNamedFeatures(),
      });
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const filledCount = [
    level, canton, theme1, theme2, theme3, institution, counterProposal,
    action, decisionQuorum, legalNormHierarchy, degreeOfRevision, legalActType,
    voteTrigger, voteTriggerActor, voteObject, author, specialTopics,
  ].filter(Boolean).length;

  return (
    <form onSubmit={handleSubmit} className="card">
      <h2 className="section-title">New Prediction</h2>
      <p style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: "1rem" }}>
        Fill in as many fields as you like — the model predicts turnout and infers
        any missing information from training-data patterns.
        {filledCount > 0 && (
          <strong> ({filledCount} field{filledCount !== 1 ? "s" : ""} specified)</strong>
        )}
      </p>

      {/* ── Core ─────────────────────────────────────────────────────────── */}
      <div className="form-row">
        <label className="form-label" style={{ flex: 2 }}>
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
          Date
          <input
            type="date"
            className="form-input"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>
      </div>

      {/* ── Level & Canton ───────────────────────────────────────────────── */}
      <div className="form-row">
        <label className="form-label">
          Level
          <select
            className="form-input"
            value={level}
            onChange={(e) => {
              setLevel(e.target.value as "national" | "cantonal" | "");
              if (e.target.value !== "cantonal") setCanton("");
            }}
          >
            <option value="">— not specified —</option>
            <option value="national">National</option>
            <option value="cantonal">Cantonal</option>
          </select>
        </label>
        {level === "cantonal" && (
          <Select label="Canton" value={canton} onChange={setCanton} options={CANTONS} />
        )}
        <Select
          label="Institution type"
          value={institution}
          onChange={setInstitution}
          options={INSTITUTIONS}
        />
      </div>

      {/* ── Theme ────────────────────────────────────────────────────────── */}
      <div className="form-row">
        <Select label="Theme 1 (primary)" value={theme1} onChange={setTheme1} options={THEMES} />
        <Select label="Theme 2 (optional)" value={theme2} onChange={setTheme2} options={THEMES} />
        <Select label="Theme 3 (optional)" value={theme3} onChange={setTheme3} options={THEMES} />
      </div>

      {/* ── Key attributes ───────────────────────────────────────────────── */}
      <div className="form-row">
        <Select label="Counter proposal" value={counterProposal} onChange={setCounterProposal} options={["No", "Yes"]} />
        <Select label="Decision quorum" value={decisionQuorum} onChange={setDecisionQuorum} options={DECISION_QUORUMS} />
        <Select label="Action" value={action} onChange={setAction} options={ACTIONS} />
      </div>

      {/* ── Advanced toggle ──────────────────────────────────────────────── */}
      <button
        type="button"
        className="btn btn-secondary"
        style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? "▲ Hide advanced fields" : "▼ Show advanced fields"}
      </button>

      {showAdvanced && (
        <div style={{ marginTop: "0.75rem" }}>
          <div className="form-row">
            <Select label="Legal act type" value={legalActType} onChange={setLegalActType} options={LEGAL_ACT_TYPES} />
            <Select label="Hierarchy of legal norm" value={legalNormHierarchy} onChange={setLegalNormHierarchy} options={LEGAL_NORM_HIERARCHIES} />
            <Select label="Degree of revision" value={degreeOfRevision} onChange={setDegreeOfRevision} options={DEGREES_OF_REVISION} />
          </div>
          <div className="form-row">
            <Select label="Vote trigger" value={voteTrigger} onChange={setVoteTrigger} options={VOTE_TRIGGERS} />
            <Select label="Vote trigger actor" value={voteTriggerActor} onChange={setVoteTriggerActor} options={VOTE_TRIGGER_ACTORS} />
            <Select label="Vote object" value={voteObject} onChange={setVoteObject} options={VOTE_OBJECTS} />
          </div>
          <div className="form-row">
            <Select label="Author of vote object" value={author} onChange={setAuthor} options={AUTHORS} />
            <Select label="Special topics" value={specialTopics} onChange={setSpecialTopics} options={SPECIAL_TOPICS} />
          </div>
        </div>
      )}

      {error && <p className="error-text">{error}</p>}

      <div className="form-actions" style={{ marginTop: "1rem" }}>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? "Predicting…" : "Predict turnout"}
        </button>
      </div>
    </form>
  );
}
