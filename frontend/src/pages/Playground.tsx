import { useEffect, useState, type FormEvent } from "react";
import { ArrowRight, Check, Copy, LoaderCircle, Sparkles, Waypoints } from "lucide-react";
import { api, ApiError } from "../api/client";
import type { ChatResponse, RouteExplanation } from "../types/api";
import { Card, ErrorNotice } from "../components/common/ui";
import { displayModel, formatCost, formatLatency, formatQuality } from "../utils/format";

const examplePrompt = "Explain binary search in simple terms.";
const DEFAULT_FLOOR = 0.85;
const DEFAULT_POLICY_LABEL = "V5 Platt Cost-Aware Router";

type DemoPrompt = { category: string; prompt: string };
const demoPrompts: DemoPrompt[] = [
  { category: "Simple factual", prompt: "What is the capital of France?" },
  { category: "Simple programming", prompt: "Write a Python function that returns the maximum number in a list." },
  { category: "Basic mathematics", prompt: "If a train travels 120 km in 2 hours, what is its average speed?" },
  { category: "Machine learning", prompt: "Explain the difference between supervised and unsupervised learning." },
  { category: "Deep learning", prompt: "Explain why convolutional neural networks are useful for image classification." },
  { category: "SQL", prompt: "Explain how an SQL JOIN works and give an example." },
  { category: "Moderate coding", prompt: "Given an array of integers, find the longest subarray whose sum is equal to K. Explain the approach and provide Java code." },
  { category: "System design", prompt: "Design a scalable URL shortener and explain the main components." },
  { category: "Distributed systems", prompt: "Explain the Saga pattern and compensating transactions in microservices." },
  { category: "ML system design", prompt: "Design a production machine learning inference service with model versioning, monitoring, rollback, and autoscaling." },
  { category: "Long technical architecture", prompt: "Design a multi-region event ingestion platform for a global commerce service. It must accept bursty traffic, preserve per-customer ordering, support replay, and provide at-least-once delivery. Describe the API, partitioning strategy, durable storage, consumer coordination, failure recovery, observability, security boundaries, and deployment topology. Explain the tradeoffs between latency, durability, consistency, and operating cost. Include a staged rollout plan and the failure scenarios that should be exercised before launch." },
  { category: "Research-style", prompt: "Compare retrieval-augmented generation and fine-tuning for adapting a language model to a changing technical knowledge base. Discuss data freshness, factual grounding, latency, cost, privacy, evaluation design, and maintenance. Give a decision framework and identify cases where a hybrid approach is justified." },
  { category: "Creative", prompt: "Write a short science-fiction story about an AI engineer debugging a distributed system." },
  { category: "Complex coding", prompt: "Design and implement an in-memory rate limiter that supports per-user limits, a sliding time window, concurrent requests, and bounded memory. Explain the data structures, thread-safety approach, time complexity, cleanup strategy, and tests for boundary conditions. Provide Python code." },
  { category: "Complex research and architecture", prompt: "Propose an architecture for serving privacy-sensitive machine-learning predictions across several regions when some regions can become disconnected. Inputs can arrive out of order and clients need reproducible results for a given model version. Describe data residency, model distribution, feature consistency, request idempotency, fallback behavior, auditability, observability, disaster recovery, and rollback. Compare at least two consistency strategies, state their failure modes, and give a migration plan from a single-region service without disrupting existing clients." },
];

type EvaluationRow = { category: string; prompt: string; explanation: RouteExplanation; result: ChatResponse };

export function Playground() {
  const [view, setView] = useState<"playground" | "evaluation">("playground");
  const [prompt, setPrompt] = useState("");
  const [qualityFloor, setQualityFloor] = useState(DEFAULT_FLOOR);
  const [policyLabel, setPolicyLabel] = useState(DEFAULT_POLICY_LABEL);
  const [quality, setQuality] = useState(DEFAULT_FLOOR);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [explanation, setExplanation] = useState<RouteExplanation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [evaluationRows, setEvaluationRows] = useState<EvaluationRow[]>([]);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [evaluationProgress, setEvaluationProgress] = useState(0);
  const [evaluationRunning, setEvaluationRunning] = useState(false);

  useEffect(() => {
    let active = true;
    api.routingConfig().then((cfg) => {
      if (!active || typeof cfg?.quality_floor !== "number") return;
      const floor = Number(cfg.quality_floor.toFixed(2));
      setQualityFloor(floor);
      if (cfg.policy_id) setPolicyLabel(cfg.policy_id);
      setQuality((prev) => (prev === DEFAULT_FLOOR || prev < floor ? floor : prev));
    }).catch(() => {
      // Keep safe V2.1 default (0.90) if routingConfig is unavailable.
    });
    return () => { active = false; };
  }, []);

  const runPrompt = async (text: string, floor: number) => {
    const route = await api.explain({ prompt: text, min_quality: floor });
    const response = await api.chat({ prompt: text, min_quality: floor });
    return { route, response };
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!prompt.trim() || loading) return;
    setLoading(true); setError(null); setResult(null); setExplanation(null);
    try {
      const { route, response } = await runPrompt(prompt.trim(), quality);
      setExplanation(route); setResult(response);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "The request could not be completed.");
    } finally { setLoading(false); }
  };

  const clear = () => { setPrompt(""); setResult(null); setExplanation(null); setError(null); };
  const copyRequestId = async () => {
    if (!result) return;
    try { await navigator.clipboard.writeText(result.request_id); setCopied(true); window.setTimeout(() => setCopied(false), 1400); }
    catch { setCopied(false); }
  };

  const runEvaluation = async () => {
    if (evaluationRunning) return;
    setEvaluationRunning(true); setEvaluationError(null); setEvaluationRows([]); setEvaluationProgress(0);
    const rows: EvaluationRow[] = [];
    try {
      for (let index = 0; index < demoPrompts.length; index += 1) {
        const item = demoPrompts[index];
        const { route, response } = await runPrompt(item.prompt, qualityFloor);
        rows.push({ category: item.category, prompt: item.prompt, explanation: route, result: response });
        setEvaluationRows([...rows]); setEvaluationProgress(index + 1);
      }
    } catch (reason) {
      setEvaluationError(reason instanceof ApiError ? reason.message : "The OOD run stopped before completion.");
    } finally { setEvaluationRunning(false); }
  };

  return <>
    <div className="mode-tabs" role="tablist" aria-label="Playground modes">
      <button role="tab" aria-selected={view === "playground"} className={view === "playground" ? "selected" : ""} onClick={() => setView("playground")}>Single prompt</button>
      <button role="tab" aria-selected={view === "evaluation"} className={view === "evaluation" ? "selected" : ""} onClick={() => setView("evaluation")}>Evaluation Mode <span>15 prompts</span></button>
    </div>
    {view === "playground" ? <>
      <div className="playground-grid">
        <Card className="prompt-card">
          <div className="card-topline"><div><div className="card-kicker"><span className="kicker-icon"><Sparkles size={16} /></span>ROUTING PLAYGROUND</div><h2>Try a prompt</h2><p className="muted-copy">The active router ({policyLabel}) scores each candidate, then applies its quality and cost policy.</p></div><span className="tag tag-neutral">Local demo</span></div>
          <form onSubmit={submit}>
            <label className="field-label" htmlFor="prompt">Prompt</label>
            <textarea id="prompt" className="prompt-input" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Enter your prompt..." rows={9} maxLength={12000} />
            <div className="textarea-footer">
              <span>{prompt.length.toLocaleString()} characters</span>
              <div>
                <button type="button" className="text-button" onClick={() => setPrompt(examplePrompt)}>Use example prompt</button>
                <button type="button" className="text-button" onClick={() => setPrompt("What is the capital of France?")} title="Prompts with factual lexical cues reach >=0.90 and route to DeepSeek">Factual (DeepSeek)</button>
                <button type="button" className="text-button" onClick={clear} disabled={loading}>Clear</button>
              </div>
            </div>
            <div className="controls-row">
              <div className="quality-control"><div className="field-label">Quality target <strong className="quality-value">{quality.toFixed(2)}</strong></div><input aria-label="Quality target" type="range" min={qualityFloor.toFixed(2)} max="1.00" step="0.01" value={quality} onChange={(event) => setQuality(Number(event.target.value))} /><div className="range-labels"><span>{qualityFloor.toFixed(2)}</span><span>1.00</span></div><p className="control-hint">{policyLabel} policy floor: {qualityFloor.toFixed(2)}</p></div>
              <button className="primary-button" type="submit" disabled={loading || !prompt.trim()}>{loading ? <><LoaderCircle size={17} className="spin" />Routing…</> : <>Route &amp; Run<ArrowRight size={17} /></>}</button>
            </div>
          </form>
        </Card>
        <div className="playground-aside">
          {error && <ErrorNotice message={error} />}
          {loading && <Card className="routing-progress"><LoaderCircle className="spin" size={20} /><div><strong>Evaluating candidate models</strong><p>Predicting quality, selecting a route, and recording request metadata.</p></div></Card>}
          {explanation && <RoutingInsights explanation={explanation} />}
          {result && <DecisionSummary result={result} quality={quality} copied={copied} onCopy={() => void copyRequestId()} explanation={explanation!} />}
          {!explanation && !loading && !error && <Card className="empty-summary"><Waypoints size={24} /><h2>Routing decision</h2><p>Candidate predictions, eligibility, selected model, and measured inference details will appear here.</p></Card>}
        </div>
      </div>
      {result && <Card className="response-card"><div className="response-head"><div><div className="card-kicker">MODEL RESPONSE</div><h2>Response</h2></div><span className="response-model">{result.provider_model}</span></div><ResponseContent text={result.response} /></Card>}
      <p className="ood-note">Predicted quality estimates are not correctness guarantees. Request telemetry stores a prompt hash and metadata, not prompt or response text.</p>
    </> : <EvaluationMode rows={evaluationRows} error={evaluationError} progress={evaluationProgress} running={evaluationRunning} onRun={() => void runEvaluation()} />}
  </>;
}

function RoutingInsights({ explanation }: { explanation: RouteExplanation }) {
  return <Card className="insights-card"><div className="section-title-row"><div><div className="card-kicker">FROZEN V2.1 PREDICTIONS</div><h2>Routing Insights</h2></div><span className="floor-tag">QUALITY FLOOR {explanation.quality_floor.toFixed(2)}</span></div>
    <div className="quality-bars">{explanation.candidates.map((candidate) => <div className="quality-bar-row" key={candidate.canonical_model}><div className="quality-bar-label"><span>{displayModel(candidate.canonical_model)}</span><strong>{candidate.predicted_quality.toFixed(3)}</strong></div><div className="quality-bar-track"><span className={candidate.eligible ? "eligible" : "below"} style={{ width: `${candidate.predicted_quality * 100}%` }} /><i style={{ left: `${explanation.quality_floor * 100}%` }} /></div><span className={`eligibility ${candidate.eligible ? "positive" : "muted"}`}>{candidate.eligible ? "Eligible" : "Below floor"}</span></div>)}</div>
    <div className={`route-outcome ${explanation.fallback ? "warning" : "positive"}`}><strong>{explanation.fallback ? "Fallback route" : "Eligible route"}</strong><span>{displayModel(explanation.selected_model)}</span></div>
    <p className="reason-copy">{explanation.routing_reason}</p>
    {explanation.fallback && (
      <div className="fallback-clarity-note">
        <strong>Policy Fallback:</strong>
        <span>All candidate predictions fell below the {explanation.quality_floor.toFixed(2)} quality floor on this prompt. The router safely defaulted to Gemini 2.5 Pro (this is a policy decision, not a provider failure).</span>
      </div>
    )}
  </Card>;
}

function DecisionSummary({ result, quality, copied, onCopy, explanation }: { result: ChatResponse; quality: number; copied: boolean; onCopy: () => void; explanation: RouteExplanation }) {
  const tone = explanation.fallback ? "warning" : "positive";
  return <Card className="decision-card"><div className="result-heading"><div><div className="card-kicker">ROUTING DECISION</div><h2>Request complete</h2></div><span className={`state-tag ${tone}`}>{explanation.fallback ? "Fallback" : "Eligible route"}</span></div>
    <div className="decision-metrics"><Metric label="Selected model" value={displayModel(result.canonical_model)} note={result.canonical_model} /><Metric label="Predicted quality" value={formatQuality(result.predicted_quality)} note="Frozen V2.1 estimate" /><Metric label="Estimated cost" value={formatCost(result.estimated_cost_usd)} note="Provider-reported usage" /><Metric label="Latency" value={formatLatency(result.latency_ms)} note="Measured request time" /></div>
    <div className="route-meta-grid"><div><span>PROVIDER MODEL</span><strong>{result.provider_model}</strong></div><div><span>INPUT TOKENS</span><strong>{result.input_tokens ?? "Unavailable"}</strong></div><div><span>OUTPUT TOKENS</span><strong>{result.output_tokens ?? "Unavailable"}</strong></div><div><span>QUALITY TARGET</span><strong>{quality.toFixed(2)}</strong></div><div className="request-id"><span>REQUEST ID</span><strong>{result.request_id}<button onClick={onCopy} aria-label="Copy request ID">{copied ? <Check size={14} /> : <Copy size={14} />}</button></strong></div></div>
    {result.model_mapping_warning && <div className="mapping-warning">{result.model_mapping_warning}</div>}
    <div className="reason-panel"><strong>Routing reason</strong><p>{result.routing_reason}</p></div>
  </Card>;
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="decision-metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function ResponseContent({ text }: { text: string }) {
  const chunks: { code: boolean; text: string; language?: string }[] = [];
  const pattern = /```([\w+-]*)\s*\n([\s\S]*?)```/g;
  let previous = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > previous) chunks.push({ code: false, text: text.slice(previous, start) });
    chunks.push({ code: true, text: match[2], language: match[1] });
    previous = start + match[0].length;
  }
  if (!chunks.length) return <div className="response-body">{text}</div>;
  if (previous < text.length) chunks.push({ code: false, text: text.slice(previous) });
  return <div className="response-body">{chunks.map((chunk, index) => chunk.code
    ? <pre key={index} className="response-code"><code>{chunk.text}</code></pre>
    : <p key={index}>{chunk.text}</p>)}</div>;
}

function EvaluationMode({ rows, error, progress, running, onRun }: { rows: EvaluationRow[]; error: string | null; progress: number; running: boolean; onRun: () => void }) {
  return <div className="evaluation-page"><Card className="evaluation-intro"><div><div className="card-kicker">QUALITATIVE GENERALIZATION CHECK</div><h2>Unseen/OOD Demo Prompts</h2><p>These 15 built-in prompts are exploratory examples, not a benchmark or accuracy evaluation. Each prompt runs through the frozen V2.1 router and current provider configuration.</p></div><button className="primary-button" onClick={onRun} disabled={running}>{running ? <><LoaderCircle size={17} className="spin" />Running {progress} / {demoPrompts.length}</> : <>Run all prompts<ArrowRight size={17} /></>}</button></Card>
    <div className="evaluation-note"><strong>Research note</strong><span>Routing decisions are produced by the frozen V2.1 model. OOD prompts are qualitative tests and are not part of the benchmark evaluation.</span></div>
    {error && <ErrorNotice message={error} />}
    <Card className="requests-card evaluation-table-card"><div className="section-title-row"><div><div className="card-kicker">15 PROMPT CATEGORIES</div><h2>OOD run results</h2></div><span className="history-count">{progress} / {demoPrompts.length} completed</span></div>
      <div className="table-scroll"><table className="evaluation-table"><thead><tr><th>Category / prompt preview</th><th>DeepSeek</th><th>Gemini Flash</th><th>Gemini Pro</th><th>Floor</th><th>Selected / fallback</th><th>Cost</th><th>Latency</th></tr></thead><tbody>
        {rows.map((row) => <tr key={row.category}><td className="ood-prompt-cell"><strong>{row.category}</strong><span>{row.prompt}</span></td>{["deepseek-v3-0324", "gemini-2.5-flash", "gemini-2.5-pro"].map((model) => <td key={model}>{row.explanation.candidates.find((candidate) => candidate.canonical_model === model)?.predicted_quality.toFixed(3) ?? "—"}</td>)}<td>{row.explanation.quality_floor.toFixed(2)}</td><td><strong>{displayModel(row.explanation.selected_model)}</strong><span className={`state-tag ${row.explanation.fallback ? "warning" : "positive"}`}>{row.explanation.fallback ? "Fallback" : "Eligible"}</span></td><td>{formatCost(row.result.estimated_cost_usd)}</td><td>{formatLatency(row.result.latency_ms)}</td></tr>)}
      </tbody></table></div>
      {!rows.length && <div className="empty-inline">Run the prompt set to see actual frozen-router predictions and mock/real inference measurements.</div>}
    </Card>
    <p className="ood-note">The API receives prompt text transiently to score and run each example. PostgreSQL telemetry stores hashes and request metadata only.</p>
  </div>;
}
