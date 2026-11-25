import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  Clock3,
  Coins,
  Gauge,
  Layers3,
  RefreshCw,
  ShieldCheck,
  TriangleAlert,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { api, ApiError } from "../api/client";
import type { AnalyticsSummary as Summary, InferenceRequestRecord, ModelAnalytics } from "../types/api";
import { displayModel, formatCost, formatLatency, formatTimestamp, modelColor } from "../utils/format";
import {
  Card,
  EmptyState,
  ErrorNotice,
  LoadingState,
  QualityBar,
  SectionTitle,
  StatusBadge,
} from "../components/common/ui";
import { TimeSeriesChart } from "../components/common/TimeSeriesChart";
import { RoutingFlowDiagram } from "../components/common/RoutingFlowDiagram";

interface AnalyticsProps {
  mode?: "overview" | "analytics";
}

const modelOptions = [
  ["deepseek-v3-0324", "DeepSeek V3"],
  ["gemini-2.5-flash", "Gemini 2.5 Flash"],
  ["gemini-2.5-pro", "Gemini 2.5 Pro"],
];

export function Analytics({ mode: _mode = "analytics" }: AnalyticsProps) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [models, setModels] = useState<ModelAnalytics[]>([]);
  const [recent, setRecent] = useState<InferenceRequestRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Table filter states
  const [filterModel, setFilterModel] = useState<string>("");
  const [filterStatus, setFilterStatus] = useState<string>("");
  const [tablePage, setTablePage] = useState<number>(0);
  const PAGE_SIZE = 5;

  const shouldReduceMotion = useReducedMotion();

  const load = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const [nextSummary, nextModels, nextRecent] = await Promise.all([
        api.summary(),
        api.models(),
        api.recent(),
      ]);
      setSummary(nextSummary);
      setModels(nextModels);
      setRecent(nextRecent);
    } catch (reason) {
      setError(
        reason instanceof ApiError ? reason.message : "Could not load analytics from telemetry."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Filtered recent requests
  const filteredRecent = useMemo(() => {
    return recent.filter((r) => {
      if (filterModel && r.canonical_model !== filterModel) return false;
      if (filterStatus && r.status !== filterStatus) return false;
      return true;
    });
  }, [recent, filterModel, filterStatus]);

  const pagedRecent = useMemo(() => {
    const start = tablePage * PAGE_SIZE;
    return filteredRecent.slice(start, start + PAGE_SIZE);
  }, [filteredRecent, tablePage]);

  if (loading) {
    return <LoadingState label="Loading analytics from PostgreSQL" />;
  }

  if (error) {
    return (
      <div className="stack">
        <ErrorNotice message={error} onRetry={() => void load()} />
        <p className="muted-copy">
          Analytics need an available telemetry database and initialized schema.
        </p>
      </div>
    );
  }

  if (!summary || summary.total_requests === 0) {
    return (
      <Card className="analytics-empty">
        <EmptyState />
        <Link className="primary-button link-button" to="/playground">
          Open Playground
          <ArrowRight size={16} />
        </Link>
      </Card>
    );
  }

  const totalModelRequests = models.reduce((sum, row) => sum + row.request_count, 0);
  const maxTotalCost = Math.max(0, ...models.map((row) => row.total_cost_usd ?? 0));
  const maxLatency = Math.max(0, ...models.map((row) => row.average_latency_ms ?? 0));

  // Determine latest active request event for routing flow diagram
  const latestRequest = recent.length > 0 ? recent[0] : null;

  return (
    <div className="stack analytics-page">
      {/* Top Toolbar */}
      <div className="analytics-toolbar">
        <span className="data-source">
          <span className="source-dot" />
          LIVE TELEMETRY <span>·</span> PostgreSQL
        </span>
        <button
          className="secondary-button compact"
          onClick={() => void load(true)}
          disabled={refreshing}
          aria-label="Refresh telemetry data"
        >
          <RefreshCw size={14} className={refreshing ? "spin" : ""} />
          Refresh data
        </button>
      </div>

      {/* KPI Cards Grid */}
      <motion.div
        className="kpi-grid"
        initial={shouldReduceMotion ? false : "hidden"}
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: { staggerChildren: 0.08 },
          },
        }}
      >
        <Kpi
          title="Total requests"
          value={summary.total_requests.toLocaleString()}
          icon={<Layers3 size={17} />}
          tone="blue"
          caption="Recorded in telemetry"
        />

        <Kpi
          title="Average cost"
          value={formatCost(summary.average_cost_usd)}
          icon={<Coins size={17} />}
          tone="mint"
          caption="Per inference request"
        />

        <Kpi
          title="Average latency"
          value={formatLatency(summary.average_latency_ms)}
          icon={<Clock3 size={17} />}
          tone="violet"
          caption="Measured provider latency"
        />

        <Kpi
          title="Error rate"
          value={`${(summary.error_rate * 100).toFixed(1)}%`}
          icon={<TriangleAlert size={17} />}
          tone={summary.error_rate > 0 ? "amber" : "mint"}
          caption="Failed provider requests"
        />

        <Kpi
          title="Fallback rate"
          value={
            summary.fallback_rate !== null && summary.fallback_rate !== undefined
              ? `${(summary.fallback_rate * 100).toFixed(1)}%`
              : "N/A"
          }
          icon={<ShieldCheck size={17} />}
          tone="blue"
          caption={
            summary.fallback_rate !== null && summary.fallback_rate !== undefined
              ? "Below 0.90 quality floor"
              : "No post-migration records"
          }
        />
      </motion.div>

      {/* Time-Series Chart */}
      <Card className="chart-card time-series-card" delay={0.1}>
        <TimeSeriesChart requests={recent} />
      </Card>

      {/* Live Routing Flow Diagram */}
      <Card className="chart-card routing-flow-panel-card" delay={0.15}>
        <SectionTitle
          title="Model Routing Pipeline"
          detail="Active routing topology and decision flow under frozen V2.1 policy (quality floor: 0.90)"
        />
        <RoutingFlowDiagram
          activeModel={latestRequest?.canonical_model}
          activeQuality={latestRequest?.predicted_quality}
          routingReason={
            latestRequest
              ? `Recent request ${latestRequest.request_id} mapped to ${displayModel(
                  latestRequest.canonical_model
                )} with predicted quality ${(
                  (latestRequest.predicted_quality ?? 0) * 100
                ).toFixed(1)}%`
              : "Awaiting next inference request to highlight active traversal."
          }
          fallback={latestRequest?.canonical_model === "gemini-2.5-pro"}
        />
      </Card>

      {/* Model Breakdown: Usage and Cost */}
      <div className="analytics-two-col">
        <Card className="chart-card" delay={0.2}>
          <SectionTitle
            title="Model usage"
            detail={`${totalModelRequests.toLocaleString()} requests across recorded models`}
          />
          <div className="usage-bars">
            {models.map((model) => {
              const share = totalModelRequests
                ? (model.request_count / totalModelRequests) * 100
                : 0;
              return (
                <div className="usage-row" key={model.canonical_model}>
                  <div className="usage-label">
                    <span
                      className="legend-dot"
                      style={{ background: modelColor(model.canonical_model) }}
                    />
                    <span>{displayModel(model.canonical_model)}</span>
                    <strong>{share.toFixed(1)}%</strong>
                  </div>
                  <div className="bar-track">
                    <span
                      style={{
                        width: `${Math.max(share, 1)}%`,
                        background: modelColor(model.canonical_model),
                      }}
                    />
                  </div>
                  <div className="usage-count">
                    {model.request_count.toLocaleString()} requests
                  </div>
                </div>
              );
            })}
          </div>
          <div className="chart-footnote">
            <span>Distribution by canonical research model</span>
            <span>
              <Activity size={13} />
              Live records
            </span>
          </div>
        </Card>

        <Card className="chart-card model-table-card" delay={0.25}>
          <SectionTitle title="Cost by model" detail="Estimated costs from provider usage" />
          <div className="model-cost-list">
            {models.map((model) => {
              const width =
                maxTotalCost > 0 ? ((model.total_cost_usd ?? 0) / maxTotalCost) * 100 : 0;
              return (
                <div className="cost-model-row" key={model.canonical_model}>
                  <div className="cost-row-top">
                    <span className="model-name">
                      <span
                        className="legend-dot"
                        style={{ background: modelColor(model.canonical_model) }}
                      />
                      {displayModel(model.canonical_model)}
                    </span>
                    <strong>{formatCost(model.total_cost_usd)}</strong>
                  </div>
                  <div className="bar-track cost-track">
                    <span
                      style={{
                        width: `${width}%`,
                        background: modelColor(model.canonical_model),
                      }}
                    />
                  </div>
                  <div className="cost-row-bottom">
                    <span>Average {formatCost(model.average_cost_usd)}</span>
                    <span>{model.request_count} requests</span>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="chart-footnote">
            <span>Total estimated cost by canonical model</span>
            <span>USD</span>
          </div>
        </Card>
      </div>

      {/* Model Breakdown: Latency and Quality */}
      <div className="analytics-two-col">
        <Card className="chart-card" delay={0.3}>
          <SectionTitle title="Latency by model" detail="Average measured request time" />
          <div className="metric-bars">
            {models.map((model) => (
              <div className="metric-bar-row" key={model.canonical_model}>
                <span className="metric-bar-name">{displayModel(model.canonical_model)}</span>
                <div className="bar-track">
                  <span
                    style={{
                      width: `${
                        maxLatency
                          ? Math.max(2, ((model.average_latency_ms ?? 0) / maxLatency) * 100)
                          : 0
                      }%`,
                      background: modelColor(model.canonical_model),
                    }}
                  />
                </div>
                <strong>{formatLatency(model.average_latency_ms)}</strong>
              </div>
            ))}
          </div>
        </Card>

        <Card className="chart-card" delay={0.35}>
          <SectionTitle
            title="Predicted quality"
            detail="Average V2.1 probability estimate, not observed correctness"
          />
          <div className="quality-list">
            {models.map((model) => (
              <div className="quality-row" key={model.canonical_model}>
                <span>{displayModel(model.canonical_model)}</span>
                <QualityBar value={model.average_predicted_quality} />
              </div>
            ))}
          </div>
          <div className="quality-disclaimer">
            <Gauge size={14} />
            This is a model estimate. The telemetry does not record benchmark correctness.
          </div>
        </Card>
      </div>

      {/* Recent Requests Table with Filter Controls & Pagination */}
      <Card className="recent-card" delay={0.4}>
        <SectionTitle
          title="Recent activity"
          detail="Latest inference requests recorded in PostgreSQL"
          action={
            <Link className="section-link" to="/requests">
              View all requests
              <ArrowRight size={14} />
            </Link>
          }
        />

        {/* Filter Bar for Recent Table */}
        <div className="filter-bar recent-filter-bar">
          <label className="filter-select">
            <span>Model</span>
            <select
              aria-label="Filter recent requests by model"
              value={filterModel}
              onChange={(e) => {
                setFilterModel(e.target.value);
                setTablePage(0);
              }}
            >
              <option value="">All models</option>
              {modelOptions.map(([val, label]) => (
                <option key={val} value={val}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-select">
            <span>Status</span>
            <select
              aria-label="Filter recent requests by status"
              value={filterStatus}
              onChange={(e) => {
                setFilterStatus(e.target.value);
                setTablePage(0);
              }}
            >
              <option value="">All statuses</option>
              <option value="success">Success</option>
              <option value="failed">Failed</option>
            </select>
          </label>

          {(filterModel || filterStatus) && (
            <button
              className="clear-filters"
              onClick={() => {
                setFilterModel("");
                setFilterStatus("");
                setTablePage(0);
              }}
            >
              <X size={14} />
              Clear
            </button>
          )}

          <span className="recent-count-label">
            {filteredRecent.length} matching record{filteredRecent.length === 1 ? "" : "s"}
          </span>
        </div>

        {recent.length ? (
          filteredRecent.length ? (
            <>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Selected model</th>
                      <th>Predicted quality</th>
                      <th>Cost</th>
                      <th>Latency</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedRecent.map((record) => (
                      <tr key={record.request_id} className="recent-table-row">
                        <td className="time-cell">{formatTimestamp(record.created_at)}</td>
                        <td>
                          <span className="table-model">
                            <span
                              className="legend-dot"
                              style={{
                                background: modelColor(record.canonical_model ?? ""),
                              }}
                            />
                            {displayModel(record.canonical_model)}
                          </span>
                        </td>
                        <td>
                          <QualityBar value={record.predicted_quality} />
                        </td>
                        <td className="numeric-cell">
                          {formatCost(record.estimated_cost_usd)}
                        </td>
                        <td className="numeric-cell">
                          {formatLatency(record.actual_latency_ms)}
                        </td>
                        <td>
                          <StatusBadge status={record.status} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {filteredRecent.length > PAGE_SIZE && (
                <div className="recent-pagination-row">
                  <span className="pagination-info">
                    Page {tablePage + 1} of{" "}
                    {Math.ceil(filteredRecent.length / PAGE_SIZE)}
                  </span>
                  <div className="pagination-controls">
                    <button
                      className="icon-button"
                      disabled={tablePage === 0}
                      onClick={() => setTablePage((p) => Math.max(0, p - 1))}
                      aria-label="Previous page"
                    >
                      ←
                    </button>
                    <button
                      className="icon-button"
                      disabled={(tablePage + 1) * PAGE_SIZE >= filteredRecent.length}
                      onClick={() => setTablePage((p) => p + 1)}
                      aria-label="Next page"
                    >
                      →
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <EmptyState
              title="No requests match filters"
              detail="Try resetting the model or status filters above."
            />
          )
        ) : (
          <EmptyState
            title="No recent requests"
            detail="Your telemetry summary contains requests, but no recent rows were returned."
          />
        )}
      </Card>
    </div>
  );
}

function Kpi({
  title,
  value,
  icon,
  tone,
  caption,
}: {
  title: string;
  value: string;
  icon: ReactNode;
  tone: string;
  caption: string;
}) {
  return (
    <div className="kpi-card">
      <div className={`kpi-icon ${tone}`}>{icon}</div>
      <span className="kpi-title">{title}</span>
      <strong className="kpi-value">{value}</strong>
      <span className="kpi-caption">{caption}</span>
    </div>
  );
}
