export interface ChatRequest {
  prompt: string;
  min_quality: number;
}

export interface RouteExplanation {
  prompt_length: number;
  quality_floor: number;
  candidates: { canonical_model: string; predicted_quality: number; eligible: boolean }[];
  selected_model: string;
  fallback: boolean;
  routing_reason: string;
}

export interface RuntimeStatus { provider_mode: "mock" | "real" | "unknown"; }

export interface ChatResponse {
  request_id: string;
  selected_model: string;
  canonical_model: string;
  predicted_quality: number;
  routing_reason: string;
  response: string;
  input_tokens: number | null;
  output_tokens: number | null;
  latency_ms: number;
  estimated_cost_usd: number | null;
  provider_model: string;
  model_mapping_warning: string | null;
  attempted_models?: string[];
  failover?: boolean;
  policy_fallback?: boolean;
}

export interface RoutingModelConfig {
  canonical_model: string;
  display_name: string;
  provider: string;
  training_average_cost_usd: number;
  relative_cost_vs_cheapest: number;
  benchmark_input_price_per_million: number;
  benchmark_output_price_per_million: number;
  provider_model: string | null;
}

export interface RoutingConfig {
  policy_id: string;
  quality_floor: number;
  fallback_model: string;
  provider_mode: "mock" | "real" | "unknown";
  telemetry_table: string;
  models: RoutingModelConfig[];
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface AnalyticsSummary {
  total_requests: number;
  average_cost_usd: number | null;
  average_latency_ms: number | null;
  error_rate: number;
  fallback_rate?: number | null;
  failover_rate?: number | null;
}

export interface ModelAnalytics {
  canonical_model: string;
  request_count: number;
  average_cost_usd: number | null;
  total_cost_usd: number | null;
  average_latency_ms: number | null;
  average_predicted_quality: number | null;
}

export interface InferenceRequestRecord {
  request_id: string;
  created_at: string;
  prompt_hash: string;
  canonical_model: string | null;
  provider_model: string | null;
  predicted_quality: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
  actual_latency_ms: number | null;
  status: "success" | "failed";
  attempted_models?: string[] | null;
  failover?: boolean | null;
  policy_fallback?: boolean | null;
}

export interface RequestPage {
  requests: InferenceRequestRecord[];
  total: number;
  limit: number;
  offset: number;
}
