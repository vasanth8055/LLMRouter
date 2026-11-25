import { useEffect, useState, type ReactNode } from "react";
import { AlertCircle, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, LoaderCircle, RefreshCw } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

export function Card({
  children,
  className = "",
  animate = true,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  animate?: boolean;
  delay?: number;
}) {
  const shouldReduceMotion = useReducedMotion();

  if (!animate || shouldReduceMotion) {
    return <section className={`card ${className}`}>{children}</section>;
  }

  return (
    <motion.section
      className={`card ${className}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.section>
  );
}

export function SectionTitle({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-title-row">
      <div>
        <h2 className="section-title">{title}</h2>
        {detail && <p className="section-detail">{detail}</p>}
      </div>
      {action}
    </div>
  );
}

export function ErrorNotice({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <motion.div
      className="notice notice-error"
      role="alert"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
    >
      <AlertCircle size={17} />
      <span>{message}</span>
      {onRetry && (
        <button className="text-button" onClick={onRetry}>
          Retry
        </button>
      )}
    </motion.div>
  );
}

export function EmptyState({
  title = "No inference data yet",
  detail = "Run a request in Playground to start building your telemetry.",
}: {
  title?: string;
  detail?: string;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <CircleHelp size={22} />
      </div>
      <h3>{title}</h3>
      <p>{detail}</p>
    </div>
  );
}

export function LoadingState({ label = "Loading telemetry" }: { label?: string }) {
  return (
    <div className="loading-state">
      <LoaderCircle className="spin" size={19} />
      <span>{label}</span>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton-loader ${className}`} aria-hidden="true" />;
}

export function StatusBadge({ status }: { status: "success" | "failed" }) {
  return (
    <span className={`status-badge ${status}`}>
      <span className="status-badge-dot" />
      {status === "success" ? "Success" : "Failed"}
    </span>
  );
}

export function QualityBar({ value }: { value: number | null }) {
  const shouldReduceMotion = useReducedMotion();
  if (value === null) return <span className="muted-value">—</span>;
  const width = Math.max(0, Math.min(100, value * 100));

  return (
    <span className="quality-cell">
      <span className="quality-track">
        {shouldReduceMotion ? (
          <span style={{ width: `${width}%` }} />
        ) : (
          <motion.span
            initial={{ width: 0 }}
            animate={{ width: `${width}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        )}
      </span>
      <strong>{(value * 100).toFixed(1)}%</strong>
    </span>
  );
}

export function Pagination({
  offset,
  limit,
  total,
  onPage,
}: {
  offset: number;
  limit: number;
  total: number;
  onPage: (next: number) => void;
}) {
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);
  return (
    <div className="pagination">
      <span>
        Showing <strong>{start}–{end}</strong> of <strong>{total}</strong>
      </span>
      <div className="pagination-controls">
        <button
          className="icon-button"
          aria-label="Previous page"
          disabled={offset <= 0}
          onClick={() => onPage(Math.max(0, offset - limit))}
        >
          <ChevronLeft size={16} />
        </button>
        <button
          className="icon-button"
          aria-label="Next page"
          disabled={offset + limit >= total}
          onClick={() => onPage(offset + limit)}
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}

export function RefreshButton({
  onClick,
  disabled = false,
}: {
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button className="secondary-button compact" onClick={onClick} disabled={disabled}>
      <RefreshCw size={14} className={disabled ? "spin" : ""} />
      Refresh
    </button>
  );
}

export function AnimatedNumber({
  value,
  prefix = "",
  suffix = "",
}: {
  value: number;
  prefix?: string;
  suffix?: string;
}) {
  const shouldReduceMotion = useReducedMotion();
  const [displayValue, setDisplayValue] = useState(value);

  useEffect(() => {
    if (shouldReduceMotion) {
      setDisplayValue(value);
      return;
    }

    let start = displayValue;
    const end = value;
    if (start === end) return;

    const duration = 600;
    const startTime = performance.now();

    const step = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * ease;
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        setDisplayValue(end);
      }
    };

    const animId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animId);
  }, [value, shouldReduceMotion]);

  const formatted =
    Number.isInteger(value)
      ? Math.round(displayValue).toLocaleString()
      : displayValue < 0.01
      ? displayValue.toFixed(6)
      : displayValue.toFixed(2);

  return (
    <span>
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}

export { Check, ChevronDown };
