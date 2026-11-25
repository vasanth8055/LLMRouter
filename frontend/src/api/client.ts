import type {
  AnalyticsSummary,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  InferenceRequestRecord,
  ModelAnalytics,
  RequestPage,
  RouteExplanation,
  RoutingConfig,
  RuntimeStatus,
} from "../types/api";

const API_BASE_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError("Could not connect to the API. Check that FastAPI is running.", 0);
  }

  if (!response.ok) {
    let message = `Request failed with HTTP ${response.status}.`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // Keep the status message if the server did not return JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  runtime: () => request<RuntimeStatus>("/v1/runtime"),
  routingConfig: () => request<RoutingConfig>("/v1/routing/config"),
  explain: (body: { prompt: string; min_quality: number }) =>
    request<RouteExplanation>("/v1/route/explain", { method: "POST", body: JSON.stringify(body) }),
  chat: (body: ChatRequest) =>
    request<ChatResponse>("/v1/chat", { method: "POST", body: JSON.stringify(body) }),
  summary: () => request<AnalyticsSummary>("/v1/analytics/summary"),
  models: () => request<ModelAnalytics[]>("/v1/analytics/models"),
  recent: () => request<InferenceRequestRecord[]>("/v1/analytics/recent"),
  requests: (params: { limit: number; offset: number; model?: string; status?: string; search?: string }) => {
    const query = new URLSearchParams({ limit: String(params.limit), offset: String(params.offset) });
    if (params.model) query.set("model", params.model);
    if (params.status) query.set("status", params.status);
    if (params.search) query.set("search", params.search);
    return request<RequestPage>(`/v1/requests?${query.toString()}`);
  },
};
