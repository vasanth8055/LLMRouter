import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  ArrowUpRight,
  FlaskConical,
  LayoutDashboard,
  ListFilter,
  Menu,
  RefreshCw,
  Settings as SettingsIcon,
  Sparkles,
  X,
} from "lucide-react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { api } from "../../api/client";

const navItems = [
  { to: "/overview", label: "Overview", icon: LayoutDashboard },
  { to: "/analytics", label: "Analytics", icon: Sparkles },
  { to: "/playground", label: "Playground", icon: FlaskConical },
  { to: "/requests", label: "Requests", icon: ListFilter },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

const pageInfo: Record<string, { title: string; description: string }> = {
  "/overview": {
    title: "Overview",
    description: "Real-time cost, quality, and latency telemetry across candidate LLMs.",
  },
  "/analytics": {
    title: "Analytics",
    description: "Understand the cost, quality estimates, and latency in your telemetry.",
  },
  "/playground": {
    title: "Playground",
    description: "Explore a prompt with the frozen V2.1 routing policy.",
  },
  "/requests": {
    title: "Requests",
    description: "Inspect safe inference metadata recorded by PostgreSQL.",
  },
  "/settings": {
    title: "Settings",
    description: "View frozen routing configuration, candidate model registry, and privacy policies.",
  },
};

export function Layout() {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [providerMode, setProviderMode] = useState<"mock" | "real" | "unknown">("unknown");
  const [checking, setChecking] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const location = useLocation();
  const shouldReduceMotion = useReducedMotion();
  const info = pageInfo[location.pathname] ?? pageInfo["/playground"];

  const checkHealth = async () => {
    setChecking(true);
    try {
      const result = await api.health();
      setApiOnline(result.status === "ok");
      const now = new Date();
      setLastUpdated(
        now.toLocaleTimeString(undefined, {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    } catch {
      setApiOnline(false);
      setProviderMode("unknown");
      setChecking(false);
      return;
    }
    try {
      setProviderMode((await api.runtime()).provider_mode);
    } catch {
      setProviderMode("unknown");
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    void checkHealth();
  }, []);

  useEffect(() => setMobileNavOpen(false), [location.pathname]);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark">
            <Activity size={20} strokeWidth={2.2} />
          </div>
          <div>
            <div className="brand-name">LLMRouter</div>
            <div className="brand-subtitle">Cost-Aware Multi-LLM Routing Platform</div>
          </div>
          <button
            className="icon-button mobile-close"
            aria-label="Close navigation"
            onClick={() => setMobileNavOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <div className="workspace-chip">
          <span className="workspace-dot" />
          <span>Local workspace</span>
          <span className="workspace-caret">⌄</span>
        </div>

        <div className="nav-section-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              {({ isActive }) => (
                <>
                  <Icon size={17} strokeWidth={1.9} />
                  <span>{label}</span>
                  {to === "/playground" && <ArrowUpRight className="nav-trailing" size={14} />}
                  {isActive && !shouldReduceMotion && (
                    <motion.span
                      layoutId="activeNavIndicator"
                      className="nav-active-indicator"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-spacer" />

        <div className="policy-card">
          <div className="policy-icon">
            <Activity size={15} />
          </div>
          <div>
            <div className="policy-eyebrow">ACTIVE POLICY</div>
            <div className="policy-name">V2.1 Frozen Router</div>
            <div className="policy-sub">Quality floor · 0.90</div>
          </div>
          <span className="policy-dot" />
        </div>

        <div className="sidebar-footer">
          <span>LLMRouter</span>
          <span>v0.1.0</span>
        </div>
      </aside>

      {mobileNavOpen && (
        <button
          className="mobile-scrim"
          aria-label="Close navigation"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMobileNavOpen(true)}
            >
              <Menu size={19} />
            </button>
            <div className="topbar-branding">
              <span className="topbar-logo-mark">
                <Activity size={15} />
              </span>
              <span className="topbar-brand-title">LLMRouter</span>
            </div>
            <span className="topbar-divider" />
            <div className="breadcrumb">
              <span>Workspace</span>
              <span className="breadcrumb-sep">/</span>
              <strong>{info.title}</strong>
            </div>
          </div>

          <div className="topbar-right">
            {lastUpdated && (
              <span className="topbar-timestamp" title="Last health check timestamp">
                Updated {lastUpdated}
              </span>
            )}

            <div
              className={`api-status ${
                apiOnline === null
                  ? "status-checking"
                  : apiOnline
                  ? "status-online"
                  : "status-offline"
              }`}
            >
              <span className={`status-dot ${checking ? "pulse-dot" : ""}`} />
              {apiOnline === null ? "Checking API" : apiOnline ? "API Online" : "API Offline"}
            </div>

            <div
              className={`provider-mode ${providerMode}`}
              title={
                providerMode === "mock"
                  ? "Deterministic local provider"
                  : providerMode === "real"
                  ? "External provider inference"
                  : "Provider mode unavailable"
              }
            >
              <span className="status-dot" />
              {providerMode === "unknown" ? "MODE UNKNOWN" : `${providerMode.toUpperCase()} MODE`}
            </div>

            <button
              className="icon-button health-refresh"
              aria-label="Retry API health check"
              title="Refresh connection status"
              onClick={() => void checkHealth()}
              disabled={checking}
            >
              <RefreshCw size={15} className={checking ? "spin" : ""} />
            </button>

            <span className="topbar-divider" />
            <div className="user-badge" title="Local development environment">
              LR
            </div>
          </div>
        </header>

        <div className="page-wrap">
          <motion.div
            className="page-heading"
            initial={shouldReduceMotion ? false : { opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div>
              <p className="eyebrow">COST-AWARE MULTI-LLM ROUTING PLATFORM</p>
              <h1>{info.title}</h1>
              <p className="page-description">{info.description}</p>
            </div>
            <div className="policy-pill">
              <span className="policy-pill-dot" />
              V2.1 Frozen Router
            </div>
          </motion.div>

          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={shouldReduceMotion ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={shouldReduceMotion ? undefined : { opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>

          <footer className="page-footer">
            <span>
              LLMRouter <span className="footer-dot">·</span> Cost-Aware Multi-LLM Routing
            </span>
            <span>Predicted quality is an estimate, not a correctness guarantee.</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
