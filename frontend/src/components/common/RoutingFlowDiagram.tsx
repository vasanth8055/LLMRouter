import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight, Cpu, Layers, ShieldCheck, Sparkles, Zap } from "lucide-react";
import { displayModel, formatQuality, modelColor } from "../../utils/format";

interface RoutingFlowDiagramProps {
  activeModel?: string | null;
  activeQuality?: number | null;
  routingReason?: string | null;
  fallback?: boolean;
  className?: string;
}

export function RoutingFlowDiagram({
  activeModel,
  activeQuality,
  routingReason,
  fallback = false,
  className = "",
}: RoutingFlowDiagramProps) {
  const shouldReduceMotion = useReducedMotion();

  const models = [
    {
      id: "deepseek-v3-0324",
      name: "DeepSeek V3",
      tier: "Low Cost",
      pricing: "$0.25 / $0.88 M",
    },
    {
      id: "gemini-2.5-flash",
      name: "Gemini 2.5 Flash",
      tier: "Mid Tier",
      pricing: "$0.30 / $2.50 M",
    },
    {
      id: "gemini-2.5-pro",
      name: "Gemini 2.5 Pro",
      tier: "High Tier (Fallback)",
      pricing: "$1.25 / $10.00 M",
    },
  ];

  const hasActiveEvent = Boolean(activeModel);

  return (
    <div className={`routing-flow-card ${className}`}>
      <div className="flow-card-header">
        <div className="flow-header-text">
          <div className="flow-badge">
            <Sparkles size={13} />
            <span>ROUTING ARCHITECTURE</span>
          </div>
          <h4>V2.1 Frozen Routing Pipeline</h4>
        </div>
        <div className="flow-status-pill">
          {hasActiveEvent ? (
            <span className="live-event-tag">
              <span className="pulse-dot" />
              Active Route: {displayModel(activeModel)}
            </span>
          ) : (
            <span className="static-event-tag">
              <span className="static-dot" />
              Static Topology
            </span>
          )}
        </div>
      </div>

      <div className="flow-canvas">
        {/* Stage 1: Incoming Request */}
        <div className="flow-node stage-input">
          <div className="node-icon input-icon">
            <Layers size={17} />
          </div>
          <div className="node-content">
            <span className="node-label">STAGE 1</span>
            <strong className="node-title">Incoming Request</strong>
            <span className="node-sub">Token stream & prompt hash</span>
          </div>
        </div>

        {/* Connector 1 */}
        <div className="flow-connector">
          <svg className="connector-svg" viewBox="0 0 40 24">
            <line x1="0" y1="12" x2="40" y2="12" stroke="rgba(255, 255, 255, 0.15)" strokeWidth="2" strokeDasharray={hasActiveEvent ? "none" : "3 3"} />
            {hasActiveEvent && !shouldReduceMotion && (
              <motion.circle
                cx="0"
                cy="12"
                r="3"
                fill="#38BDF8"
                animate={{ cx: [0, 40] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
              />
            )}
          </svg>
        </div>

        {/* Stage 2: Quality Predictor */}
        <div className="flow-node stage-predictor">
          <div className="node-icon predictor-icon">
            <Cpu size={17} />
          </div>
          <div className="node-content">
            <span className="node-label">STAGE 2</span>
            <strong className="node-title">Quality Predictor</strong>
            <span className="node-sub">TF-IDF + LR Classifier</span>
            {activeQuality !== null && activeQuality !== undefined && (
              <span className="node-score-tag">Score: {formatQuality(activeQuality)}</span>
            )}
          </div>
        </div>

        {/* Connector 2 */}
        <div className="flow-connector">
          <svg className="connector-svg" viewBox="0 0 40 24">
            <line x1="0" y1="12" x2="40" y2="12" stroke="rgba(255, 255, 255, 0.15)" strokeWidth="2" strokeDasharray={hasActiveEvent ? "none" : "3 3"} />
            {hasActiveEvent && !shouldReduceMotion && (
              <motion.circle
                cx="0"
                cy="12"
                r="3"
                fill="#38BDF8"
                animate={{ cx: [0, 40] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "linear", delay: 0.4 }}
              />
            )}
          </svg>
        </div>

        {/* Stage 3: Decision Policy Gate */}
        <div className="flow-node stage-gate">
          <div className="node-icon gate-icon">
            <ShieldCheck size={17} />
          </div>
          <div className="node-content">
            <span className="node-label">STAGE 3</span>
            <strong className="node-title">Policy Decision</strong>
            <span className="node-sub">Floor $\tau \ge 0.90$ · Cost Min</span>
          </div>
        </div>

        {/* Connector 3: Fan out */}
        <div className="flow-connector fan-out">
          <svg className="connector-svg" viewBox="0 0 40 24">
            <line x1="0" y1="12" x2="40" y2="12" stroke="rgba(255, 255, 255, 0.15)" strokeWidth="2" />
            {hasActiveEvent && !shouldReduceMotion && (
              <motion.circle
                cx="0"
                cy="12"
                r="3"
                fill="#2DD4BF"
                animate={{ cx: [0, 40] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "linear", delay: 0.8 }}
              />
            )}
          </svg>
        </div>

        {/* Stage 4: Candidate Models Group */}
        <div className="flow-models-group">
          {models.map((m) => {
            const isSelected = activeModel === m.id;
            return (
              <div
                key={m.id}
                className={`flow-model-card ${isSelected ? "selected-route" : ""}`}
                style={{
                  borderLeftColor: isSelected ? modelColor(m.id) : undefined,
                }}
              >
                <div className="model-card-top">
                  <span
                    className="model-dot"
                    style={{ background: modelColor(m.id) }}
                  />
                  <strong className="model-name-text">{m.name}</strong>
                  {isSelected && (
                    <span className="selected-tag">
                      {fallback ? "Fallback Selected" : "Selected"}
                    </span>
                  )}
                </div>
                <div className="model-card-details">
                  <span className="model-tier">{m.tier}</span>
                  <span className="model-pricing">{m.pricing}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Connector 4 */}
        <div className="flow-connector">
          <ArrowRight size={16} className="terminal-arrow" />
        </div>

        {/* Stage 5: Response */}
        <div className="flow-node stage-response">
          <div className="node-icon response-icon">
            <Zap size={17} />
          </div>
          <div className="node-content">
            <span className="node-label">OUTPUT</span>
            <strong className="node-title">Client Response</strong>
            <span className="node-sub">Low-cost validated output</span>
          </div>
        </div>
      </div>

      {routingReason && (
        <div className="flow-reason-footer">
          <span className="reason-label">Routing Decision:</span>
          <p className="reason-text">{routingReason}</p>
        </div>
      )}
    </div>
  );
}
