import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Search, X, FileText, Hash } from "lucide-react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { InferenceRequestRecord, RequestPage } from "../types/api";
import { displayModel, formatCost, formatLatency, formatQuality, formatTimestamp } from "../utils/format";
import { Card, EmptyState, ErrorNotice, LoadingState, Pagination, QualityBar, SectionTitle, StatusBadge } from "../components/common/ui";

const PAGE_SIZE = 10;
const modelOptions = [
  ["deepseek-v3-0324", "DeepSeek V3"],
  ["gemini-2.5-flash", "Gemini 2.5 Flash"],
  ["gemini-2.5-pro", "Gemini 2.5 Pro"],
];

export function Requests() {
  const [page, setPage] = useState<RequestPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [model, setModel] = useState("");
  const [status, setStatus] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<InferenceRequestRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setPage(await api.requests({ limit: PAGE_SIZE, offset, model: model || undefined, status: status || undefined, search: search || undefined }));
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Could not load request history.");
    } finally {
      setLoading(false);
    }
  }, [offset, model, status, search]);

  useEffect(() => { void load(); }, [load]);

  const submitSearch = (event: FormEvent) => { event.preventDefault(); setOffset(0); setSearch(searchInput.trim()); };
  const clearFilters = () => { setSearchInput(""); setSearch(""); setModel(""); setStatus(""); setOffset(0); };
  const hasFilters = Boolean(search || model || status);

  return <div className="stack requests-page">
    <Card className="requests-card">
      <SectionTitle title="Request history" detail="Safe telemetry metadata · prompt and response text are not stored" action={<span className="history-count">{page ? `${page.total.toLocaleString()} records` : ""}</span>} />
      <div className="filter-bar">
        <form className="search-field" onSubmit={submitSearch}><Search size={16} /><input aria-label="Search request ID" placeholder="Search request ID…" value={searchInput} onChange={(event) => setSearchInput(event.target.value)} /><button type="submit">Search</button></form>
        <label className="filter-select"><span>Model</span><select aria-label="Filter by model" value={model} onChange={(event) => { setModel(event.target.value); setOffset(0); }}><option value="">All models</option>{modelOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="filter-select"><span>Status</span><select aria-label="Filter by status" value={status} onChange={(event) => { setStatus(event.target.value); setOffset(0); }}><option value="">All statuses</option><option value="success">Success</option><option value="failed">Failed</option></select></label>
        {hasFilters && <button className="clear-filters" onClick={clearFilters}><X size={14} />Clear</button>}
      </div>
      {loading ? <LoadingState label="Loading requests from PostgreSQL" /> : error ? <ErrorNotice message={error} onRetry={() => void load()} /> : !page?.requests.length ? <EmptyState title={hasFilters ? "No matching requests" : "No inference data yet"} detail={hasFilters ? "Try broadening your filters." : "Run a request in Playground to start building your request history."} /> : <>
        <div className="table-scroll requests-table-scroll"><table><thead><tr><th>Timestamp</th><th>Request ID</th><th>Canonical model</th><th>Provider model</th><th>Predicted quality</th><th>Input</th><th>Output</th><th>Estimated cost</th><th>Latency</th><th>Status</th></tr></thead><tbody>
          {page.requests.map((record) => <tr key={record.request_id} className="clickable-row" tabIndex={0} onClick={() => setSelected(record)} onKeyDown={(event) => { if (event.key === "Enter") setSelected(record); }} aria-label={`View request ${record.request_id}`}>
            <td className="time-cell">{formatTimestamp(record.created_at)}</td><td className="request-id-cell">{record.request_id}</td><td>{displayModel(record.canonical_model)}</td><td className="provider-model-cell">{record.provider_model ?? "—"}</td><td><QualityBar value={record.predicted_quality} /></td><td className="numeric-cell">{record.input_tokens ?? "—"}</td><td className="numeric-cell">{record.output_tokens ?? "—"}</td><td className="numeric-cell">{formatCost(record.estimated_cost_usd)}</td><td className="numeric-cell">{formatLatency(record.actual_latency_ms)}</td><td><StatusBadge status={record.status} /></td>
          </tr>)}
        </tbody></table></div>
        <Pagination offset={page.offset} limit={page.limit} total={page.total} onPage={setOffset} />
      </>}
    </Card>
    <div className="privacy-inline"><FileText size={15} /><span>Request history contains only routing and usage metadata. Raw prompt and response content is intentionally unavailable.</span></div>
    {selected && <RequestDrawer record={selected} onClose={() => setSelected(null)} />}
  </div>;
}

function RequestDrawer({ record, onClose }: { record: InferenceRequestRecord; onClose: () => void }) {
  return <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <aside className="request-drawer" role="dialog" aria-modal="true" aria-label="Request details">
      <div className="drawer-header"><div><p className="eyebrow">REQUEST DETAILS</p><h2>Inference record</h2></div><button className="icon-button" aria-label="Close details" onClick={onClose}><X size={18} /></button></div>
      <div className="drawer-status-row"><StatusBadge status={record.status} /><span>{formatTimestamp(record.created_at)}</span></div>
      <div className="drawer-section"><span className="drawer-label">REQUEST ID</span><strong className="drawer-mono">{record.request_id}</strong></div>
      <div className="drawer-section"><span className="drawer-label">ROUTING MODELS</span><div className="drawer-models"><div><span>Canonical</span><strong>{displayModel(record.canonical_model)}</strong></div><div><span>Provider</span><strong>{record.provider_model ?? "Unavailable"}</strong></div></div></div>
      <div className="drawer-stats"><DrawerStat label="Predicted quality" value={formatQuality(record.predicted_quality)} /><DrawerStat label="Latency" value={formatLatency(record.actual_latency_ms)} /><DrawerStat label="Estimated cost" value={formatCost(record.estimated_cost_usd)} /><DrawerStat label="Tokens in / out" value={`${record.input_tokens ?? "—"} / ${record.output_tokens ?? "—"}`} /></div>
      <div className="drawer-section prompt-hash"><div className="drawer-label"><Hash size={13} />PROMPT HASH</div><code>{record.prompt_hash}</code></div>
      <div className="drawer-privacy"><span><FileText size={15} /></span><div><strong>Prompt and response not stored</strong><p>This record contains only the prompt hash and routing metadata.</p></div></div>
      <Link className="drawer-link" to="/analytics">View analytics</Link>
    </aside>
  </div>;
}

function DrawerStat({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
