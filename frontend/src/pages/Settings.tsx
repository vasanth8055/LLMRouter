import { useEffect, useState } from "react";
import { Shield, Cpu, Database, Server, Key, Lock, CheckCircle2 } from "lucide-react";
import { api } from "../api/client";
import type { RoutingConfig, RoutingModelConfig } from "../types/api";
import { Card, SectionTitle } from "../components/common/ui";

const FALLBACK_MODELS: RoutingModelConfig[] = [
  {
    canonical_model: "deepseek-v3-0324",
    display_name: "DeepSeek V3",
    provider: "deepseek",
    training_average_cost_usd: 0.000586,
    relative_cost_vs_cheapest: 1.0,
    benchmark_input_price_per_million: 0.25,
    benchmark_output_price_per_million: 0.88,
    provider_model: null,
  },
  {
    canonical_model: "gemini-2.5-flash",
    display_name: "Gemini 2.5 Flash",
    provider: "gemini",
    training_average_cost_usd: 0.00361,
    relative_cost_vs_cheapest: 6.16,
    benchmark_input_price_per_million: 0.3,
    benchmark_output_price_per_million: 2.5,
    provider_model: null,
  },
  {
    canonical_model: "gemini-2.5-pro",
    display_name: "Gemini 2.5 Pro",
    provider: "gemini",
    training_average_cost_usd: 0.038847,
    relative_cost_vs_cheapest: 66.3,
    benchmark_input_price_per_million: 1.25,
    benchmark_output_price_per_million: 10.0,
    provider_model: null,
  },
];

const MODEL_DESCRIPTIONS: Record<string, { dot: string; tagClass: string; tag: string; role: string; desc: string }> = {
  "deepseek-v3-0324": {
    dot: "deepseek",
    tagClass: "lowest-cost",
    tag: "Lowest Cost",
    role: "Primary Cost Saver",
    desc: "Lowest-cost benchmark candidate for queries scoring at or above the 0.90 predicted-quality floor.",
  },
  "gemini-2.5-flash": {
    dot: "flash",
    tagClass: "fast",
    tag: "Mid Tier",
    role: "Balanced Candidate",
    desc: "Mid-tier candidate selected when DeepSeek scores below floor and Flash meets the 0.90 constraint.",
  },
  "gemini-2.5-pro": {
    dot: "pro",
    tagClass: "frontier",
    tag: "Fallback Anchor",
    role: "Safety Fallback",
    desc: "Highest-cost benchmark model and safety fallback when cheaper candidates score below 0.90 or fail.",
  },
};

export function Settings() {
  const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const [config, setConfig] = useState<RoutingConfig | null>(null);

  useEffect(() => {
    let active = true;
    api
      .routingConfig()
      .then((next) => {
        if (active) setConfig(next);
      })
      .catch(() => {
        // Keep verified offline fallback metadata when API is unreachable.
      });
    return () => {
      active = false;
    };
  }, []);

  const models = config?.models ?? FALLBACK_MODELS;
  const qualityFloor = config?.quality_floor ?? 0.85;
  const telemetryTable = config?.telemetry_table ?? "inference_requests";

  return (
    <div className="stack settings-page">
      <div className="settings-grid">
        <Card className="settings-card" delay={0.05}>
          <SectionTitle
            title="Active Routing Policy"
            detail="Frozen multi-LLM optimization parameters and constraints"
          />
          <div className="settings-specs">
            <div className="spec-item">
              <span className="spec-label">Policy Identifier</span>
              <strong className="spec-value spec-highlight">
                {config?.policy_id ?? "V2.1 Cost-Aware Router"}
              </strong>
              <p className="spec-detail">Calibrated offline on MMLU-Pro benchmark distributions.</p>
            </div>

            <div className="spec-item">
              <span className="spec-label">Quality Floor (Constraint)</span>
              <strong className="spec-value">{qualityFloor.toFixed(2)}</strong>
              <p className="spec-detail">
                Strict minimum predicted quality target. Any candidate below {qualityFloor.toFixed(2)} is marked ineligible.
              </p>
            </div>

            <div className="spec-item">
              <span className="spec-label">Objective Function</span>
              <strong className="spec-value">Cost Minimization</strong>
              <p className="spec-detail">
                Selects the model with the lowest training benchmark cost among eligible candidates.
              </p>
            </div>

            <div className="spec-item">
              <span className="spec-label">Fallback Policy</span>
              <strong className="spec-value">Gemini 2.5 Pro</strong>
              <p className="spec-detail">
                If no candidate satisfies the {qualityFloor.toFixed(2)} quality threshold, the router defaults to {config?.fallback_model ?? "gemini-2.5-pro"} (plus runtime provider failover across eligible models).
              </p>
            </div>
          </div>
        </Card>

        <Card className="settings-card" delay={0.1}>
          <SectionTitle
            title="Privacy & Data Governance"
            detail="Built-in security guarantees for enterprise telemetry"
          />
          <div className="privacy-features">
            <div className="privacy-feature-card">
              <div className="feature-icon">
                <Lock size={18} />
              </div>
              <div>
                <strong>SHA-256 Prompt Hashing</strong>
                <p>
                  Prompts are transformed to a 64-character SHA-256 digest before recording. Raw prompt strings are never written to the telemetry database.
                </p>
              </div>
            </div>

            <div className="privacy-feature-card">
              <div className="feature-icon">
                <Shield size={18} />
              </div>
              <div>
                <strong>Zero Completion Storage</strong>
                <p>
                  Generated completions and responses are transmitted directly to the caller. Telemetry captures only token counts, cost estimates, and measured latency.
                </p>
              </div>
            </div>

            <div className="privacy-feature-card">
              <div className="feature-icon">
                <Key size={18} />
              </div>
              <div>
                <strong>Credential Isolation</strong>
                <p>
                  Provider API keys reside exclusively in backend environment variables. No client-side exposure of API secrets.
                </p>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Card className="settings-card" delay={0.15}>
        <SectionTitle
          title="Candidate Model Registry"
          detail="Training-average relative costs and benchmark token pricing used by the frozen V2.1 router"
        />
        <div className="models-registry-grid">
          {models.map((m) => {
            const meta = MODEL_DESCRIPTIONS[m.canonical_model] ?? {
              dot: "deepseek",
              tagClass: "fast",
              tag: m.provider,
              role: "Candidate",
              desc: "Configured routing candidate.",
            };
            const relLabel =
              m.relative_cost_vs_cheapest <= 1.01
                ? "1.0x (Baseline)"
                : `~${m.relative_cost_vs_cheapest.toFixed(1)}x`;
            return (
              <div className="registry-card" key={m.canonical_model}>
                <div className="registry-head">
                  <span className={`registry-dot ${meta.dot}`} />
                  <div>
                    <strong className="registry-title">{m.display_name}</strong>
                    <span className="registry-id">{m.canonical_model}</span>
                  </div>
                  <span className={`registry-tag ${meta.tagClass}`}>{meta.tag}</span>
                </div>
                <p className="registry-desc">{meta.desc}</p>
                <div className="registry-stats">
                  <div>
                    <span>Relative Cost</span>
                    <strong>{relLabel}</strong>
                  </div>
                  <div>
                    <span>Benchmark $/1M (In / Out)</span>
                    <strong>
                      ${m.benchmark_input_price_per_million.toFixed(2)} / $
                      {m.benchmark_output_price_per_million.toFixed(2)}
                    </strong>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <Card className="settings-card" delay={0.2}>
        <SectionTitle
          title="Runtime Environment"
          detail="Current backend connection and telemetry configuration"
        />
        <div className="runtime-grid">
          <div className="runtime-item">
            <span className="runtime-label">
              <Server size={14} /> Backend API Endpoint
            </span>
            <code className="runtime-val">{apiUrl}</code>
          </div>
          <div className="runtime-item">
            <span className="runtime-label">
              <Database size={14} /> Telemetry Store
            </span>
            <div className="runtime-status-val">
              <CheckCircle2 size={15} className="text-mint" />
              <span>PostgreSQL Schema ({telemetryTable} table)</span>
            </div>
          </div>
          <div className="runtime-item">
            <span className="runtime-label">
              <Cpu size={14} /> Model Inference Engine
            </span>
            <span className="runtime-val">FastAPI + Scikit-Learn TF-IDF Predictor</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
