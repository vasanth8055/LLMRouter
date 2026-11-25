import { useId, useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Activity, Clock3, Coins, Layers3 } from "lucide-react";
import type { InferenceRequestRecord } from "../../types/api";
import { formatCost, formatLatency, formatTimestamp } from "../../utils/format";

type MetricType = "volume" | "cost" | "latency";
type TimeRange = "24h" | "7d" | "all";

interface TimeSeriesChartProps {
  requests: InferenceRequestRecord[];
  className?: string;
}

interface DataPoint {
  timestamp: Date;
  label: string;
  count: number;
  cost: number;
  latency: number;
}

export function TimeSeriesChart({ requests, className = "" }: TimeSeriesChartProps) {
  const [metric, setMetric] = useState<MetricType>("volume");
  const [range, setRange] = useState<TimeRange>("all");
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const chartId = useId();
  const shouldReduceMotion = useReducedMotion();

  // Filter requests by range
  const filteredRequests = useMemo(() => {
    if (!requests.length) return [];
    if (range === "all") return requests;

    const now = Date.now();
    const cutoff = range === "24h" ? now - 24 * 3600 * 1000 : now - 7 * 24 * 3600 * 1000;
    return requests.filter((r) => new Date(r.created_at).getTime() >= cutoff);
  }, [requests, range]);

  // Aggregate into chronological data points
  const dataPoints = useMemo<DataPoint[]>(() => {
    if (!filteredRequests.length) return [];

    // Sort ascending by creation time
    const sorted = [...filteredRequests].sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );

    // If fewer than 15 requests, map each request directly as a point
    if (sorted.length <= 15) {
      return sorted.map((r) => ({
        timestamp: new Date(r.created_at),
        label: formatTimestamp(r.created_at),
        count: 1,
        cost: r.estimated_cost_usd ?? 0,
        latency: r.actual_latency_ms ?? 0,
      }));
    }

    // Group into hourly or daily buckets
    const bucketMap = new Map<string, { count: number; cost: number; latencySum: number; lastDate: Date }>();

    for (const r of sorted) {
      const d = new Date(r.created_at);
      // Group by hour if 24h, else by date
      const key = range === "24h" 
        ? `${d.toISOString().slice(0, 13)}:00` 
        : d.toISOString().slice(0, 10);

      const existing = bucketMap.get(key);
      if (existing) {
        existing.count += 1;
        existing.cost += r.estimated_cost_usd ?? 0;
        existing.latencySum += r.actual_latency_ms ?? 0;
        existing.lastDate = d;
      } else {
        bucketMap.set(key, {
          count: 1,
          cost: r.estimated_cost_usd ?? 0,
          latencySum: r.actual_latency_ms ?? 0,
          lastDate: d,
        });
      }
    }

    return Array.from(bucketMap.entries()).map(([key, val]) => ({
      timestamp: val.lastDate,
      label: key,
      count: val.count,
      cost: val.cost,
      latency: val.count ? val.latencySum / val.count : 0,
    }));
  }, [filteredRequests, range]);

  // Compute SVG dimensions and scale
  const width = 720;
  const height = 220;
  const padding = { top: 20, right: 25, bottom: 35, left: 55 };
  const graphWidth = width - padding.left - padding.right;
  const graphHeight = height - padding.top - padding.bottom;

  const values = useMemo(() => {
    return dataPoints.map((p) => {
      if (metric === "cost") return p.cost;
      if (metric === "latency") return p.latency;
      return p.count;
    });
  }, [dataPoints, metric]);

  const maxValue = useMemo(() => {
    if (!values.length) return 1;
    const max = Math.max(...values);
    return max <= 0 ? 1 : max * 1.15; // 15% headroom
  }, [values]);

  const points = useMemo(() => {
    if (dataPoints.length === 0) return [];
    if (dataPoints.length === 1) {
      return [{ x: padding.left + graphWidth / 2, y: padding.top + graphHeight / 2 }];
    }
    return dataPoints.map((_, i) => {
      const x = padding.left + (i / (dataPoints.length - 1)) * graphWidth;
      const y = padding.top + graphHeight - (values[i] / maxValue) * graphHeight;
      return { x, y };
    });
  }, [dataPoints, values, maxValue, graphWidth, graphHeight, padding.left, padding.top]);

  // Construct SVG paths
  const linePath = useMemo(() => {
    if (points.length < 2) return "";
    return points.reduce((path, pt, i) => `${path} ${i === 0 ? "M" : "L"} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`, "");
  }, [points]);

  const areaPath = useMemo(() => {
    if (points.length < 2) return "";
    const firstX = points[0].x.toFixed(1);
    const lastX = points[points.length - 1].x.toFixed(1);
    const bottomY = (padding.top + graphHeight).toFixed(1);
    return `${linePath} L ${lastX} ${bottomY} L ${firstX} ${bottomY} Z`;
  }, [points, linePath, padding.top, graphHeight]);

  const formatYValue = (val: number) => {
    if (metric === "cost") return `$${val.toFixed(4)}`;
    if (metric === "latency") return `${Math.round(val)}ms`;
    return Math.round(val).toString();
  };

  const activePoint = hoverIndex !== null && points[hoverIndex] ? points[hoverIndex] : null;
  const activeData = hoverIndex !== null && dataPoints[hoverIndex] ? dataPoints[hoverIndex] : null;

  return (
    <div className={`chart-container-root ${className}`}>
      <div className="chart-header-row">
        <div className="chart-meta">
          <div className="chart-title-lockup">
            <Activity size={16} className="chart-title-icon" />
            <h3 className="chart-title">Telemetry Activity Trend</h3>
          </div>
          <span className="chart-subtitle">
            {filteredRequests.length} recorded request{filteredRequests.length === 1 ? "" : "s"} in range
          </span>
        </div>

        <div className="chart-controls-group">
          {/* Metric Selector */}
          <div className="chart-pill-selector" role="radiogroup" aria-label="Metric toggle">
            <button
              type="button"
              className={`pill-btn ${metric === "volume" ? "active" : ""}`}
              onClick={() => setMetric("volume")}
              aria-label="Show Request Volume"
            >
              <Layers3 size={13} />
              <span>Volume</span>
            </button>
            <button
              type="button"
              className={`pill-btn ${metric === "cost" ? "active" : ""}`}
              onClick={() => setMetric("cost")}
              aria-label="Show Estimated Cost"
            >
              <Coins size={13} />
              <span>Cost</span>
            </button>
            <button
              type="button"
              className={`pill-btn ${metric === "latency" ? "active" : ""}`}
              onClick={() => setMetric("latency")}
              aria-label="Show Latency"
            >
              <Clock3 size={13} />
              <span>Latency</span>
            </button>
          </div>

          {/* Time Range Selector */}
          <div className="chart-range-selector" role="radiogroup" aria-label="Time range toggle">
            <button
              type="button"
              className={`range-btn ${range === "24h" ? "active" : ""}`}
              onClick={() => setRange("24h")}
            >
              24h
            </button>
            <button
              type="button"
              className={`range-btn ${range === "7d" ? "active" : ""}`}
              onClick={() => setRange("7d")}
            >
              7d
            </button>
            <button
              type="button"
              className={`range-btn ${range === "all" ? "active" : ""}`}
              onClick={() => setRange("all")}
            >
              All
            </button>
          </div>
        </div>
      </div>

      {/* Main Chart Area */}
      {dataPoints.length === 0 ? (
        <div className="chart-empty-state">
          <Activity size={24} className="muted-icon" />
          <p className="empty-title">No telemetry activity recorded for this period</p>
          <p className="empty-detail">
            Inference requests executed in the Playground will populate volume, cost, and latency trends.
          </p>
        </div>
      ) : (
        <div className="chart-svg-wrapper">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="chart-svg"
            onMouseLeave={() => setHoverIndex(null)}
          >
            <defs>
              <linearGradient id={`chart-grad-${chartId}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.32" />
                <stop offset="60%" stopColor="#0EA5E9" stopOpacity="0.08" />
                <stop offset="100%" stopColor="#0369A1" stopOpacity="0.00" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
              const y = padding.top + graphHeight * (1 - pct);
              const labelVal = maxValue * pct;
              return (
                <g key={idx} className="chart-grid-line-group">
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={width - padding.right}
                    y2={y}
                    stroke="rgba(255, 255, 255, 0.06)"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={padding.left - 8}
                    y={y + 3}
                    textAnchor="end"
                    className="chart-axis-label"
                  >
                    {formatYValue(labelVal)}
                  </text>
                </g>
              );
            })}

            {/* Area Fill */}
            {areaPath && (
              <motion.path
                d={areaPath}
                fill={`url(#chart-grad-${chartId})`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: shouldReduceMotion ? 0 : 0.5 }}
              />
            )}

            {/* Stroke Line */}
            {linePath && (
              <motion.path
                d={linePath}
                fill="none"
                stroke="#38BDF8"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
                initial={shouldReduceMotion ? { pathLength: 1 } : { pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: shouldReduceMotion ? 0 : 0.8, ease: "easeOut" }}
              />
            )}

            {/* Interactive Data points & Hover columns */}
            {points.map((pt, i) => (
              <g key={i}>
                {/* Invisible hover bar */}
                <rect
                  x={pt.x - (graphWidth / points.length) / 2}
                  y={padding.top}
                  width={graphWidth / points.length}
                  height={graphHeight}
                  fill="transparent"
                  onMouseEnter={() => setHoverIndex(i)}
                  className="chart-hover-target"
                />

                {/* Point dot */}
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={hoverIndex === i ? 5 : 3}
                  className={`chart-data-dot ${hoverIndex === i ? "active" : ""}`}
                />
              </g>
            ))}

            {/* Active Vertical Crosshair */}
            {activePoint && (
              <g className="chart-crosshair">
                <line
                  x1={activePoint.x}
                  y1={padding.top}
                  x2={activePoint.x}
                  y2={padding.top + graphHeight}
                  stroke="#38BDF8"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  opacity={0.6}
                />
                <circle
                  cx={activePoint.x}
                  cy={activePoint.y}
                  r={6}
                  fill="#0369A1"
                  stroke="#38BDF8"
                  strokeWidth={2}
                />
              </g>
            )}

            {/* X-axis labels */}
            {points.length > 1 && (
              <>
                <text
                  x={points[0].x}
                  y={height - 8}
                  textAnchor="start"
                  className="chart-axis-label-x"
                >
                  {dataPoints[0].label.slice(0, 16)}
                </text>
                <text
                  x={points[points.length - 1].x}
                  y={height - 8}
                  textAnchor="end"
                  className="chart-axis-label-x"
                >
                  {dataPoints[dataPoints.length - 1].label.slice(0, 16)}
                </text>
              </>
            )}
          </svg>

          {/* Floating Tooltip */}
          {activePoint && activeData && (
            <motion.div
              className="chart-tooltip"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.15 }}
              style={{
                left: `${(activePoint.x / width) * 100}%`,
                top: `${(activePoint.y / height) * 100}%`,
              }}
            >
              <div className="tooltip-time">{activeData.label}</div>
              <div className="tooltip-value-row">
                <span className="tooltip-label">
                  {metric === "volume" ? "Volume:" : metric === "cost" ? "Cost:" : "Latency:"}
                </span>
                <strong className="tooltip-value">
                  {metric === "volume"
                    ? `${activeData.count} request${activeData.count === 1 ? "" : "s"}`
                    : metric === "cost"
                    ? formatCost(activeData.cost)
                    : formatLatency(activeData.latency)}
                </strong>
              </div>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}
