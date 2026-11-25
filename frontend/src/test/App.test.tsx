import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { AppRoutes } from "../App";
import type { AnalyticsSummary, ChatResponse, InferenceRequestRecord, ModelAnalytics, RouteExplanation } from "../types/api";

const summary: AnalyticsSummary = { total_requests: 4, average_cost_usd: 0.0012, average_latency_ms: 142, error_rate: 0.25 };
const models: ModelAnalytics[] = [{
  canonical_model: "gemini-2.5-flash", request_count: 4, average_cost_usd: 0.0012,
  total_cost_usd: 0.0048, average_latency_ms: 142, average_predicted_quality: 0.913,
}];
const record: InferenceRequestRecord = {
  request_id: "req-test-123", created_at: "2026-02-01T12:00:00Z", prompt_hash: "a".repeat(64),
  canonical_model: "gemini-2.5-flash", provider_model: "gemini-3.8-flash", predicted_quality: 0.913,
  input_tokens: 15, output_tokens: 27, estimated_cost_usd: 0.0012, actual_latency_ms: 142, status: "success",
};
const chatResponse: ChatResponse = {
  request_id: "req-play-001", selected_model: "gemini-2.5-flash", canonical_model: "gemini-2.5-flash",
  predicted_quality: 0.9134, routing_reason: "Selected the lowest training-cost model with predicted quality at least 0.90 under the frozen V2.1 policy.",
  response: "Binary search repeatedly halves a sorted list until it finds the target.", input_tokens: 9,
  output_tokens: 15, latency_ms: 42.5, estimated_cost_usd: 0.00068, provider_model: "gemini-3.8-flash",
  model_mapping_warning: "Live API model differs from frozen benchmark identity.",
};
const routeExplanation: RouteExplanation = {
  prompt_length: 24, quality_floor: 0.9,
  candidates: [
    { canonical_model: "deepseek-v3-0324", predicted_quality: 0.82, eligible: false },
    { canonical_model: "gemini-2.5-flash", predicted_quality: 0.9134, eligible: true },
    { canonical_model: "gemini-2.5-pro", predicted_quality: 0.94, eligible: true },
  ],
  selected_model: "gemini-2.5-flash", fallback: false,
  routing_reason: "Selected the lowest training-cost model with predicted quality at least 0.90 under the frozen V2.1 policy.",
};

function renderAt(path: string) {
  return render(<MemoryRouter initialEntries={[path]}><AppRoutes /></MemoryRouter>);
}

beforeEach(() => {
  vi.spyOn(api, "health").mockResolvedValue({ status: "ok", service: "llmrouter" });
  vi.spyOn(api, "runtime").mockResolvedValue({ provider_mode: "mock" });
  vi.spyOn(api, "explain").mockResolvedValue(routeExplanation);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("dashboard routes and API flows", () => {
  it("renders the three navigation routes", async () => {
    renderAt("/playground");
    expect(await screen.findByRole("heading", { name: "Playground" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Analytics/ })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Requests/ })).toBeInTheDocument();
  });

  it("posts the prompt and shows the actual routing response", async () => {
    vi.spyOn(api, "chat").mockResolvedValue(chatResponse);
    const user = userEvent.setup();
    renderAt("/playground");
    await user.type(screen.getByLabelText("Prompt"), "Explain binary search.");
    await user.click(screen.getByRole("button", { name: /Route & Run/ }));
    expect(await screen.findByText("Request complete")).toBeInTheDocument();
    expect(api.chat).toHaveBeenCalledWith({ prompt: "Explain binary search.", min_quality: 0.85 });
    expect(api.explain).toHaveBeenCalledWith({ prompt: "Explain binary search.", min_quality: 0.85 });
    expect(screen.getAllByText("gemini-3.8-flash").length).toBeGreaterThan(1);
    expect(screen.getAllByText(chatResponse.routing_reason).length).toBeGreaterThan(1);
    expect(screen.getByText(chatResponse.response)).toBeInTheDocument();
    expect(screen.getByText(chatResponse.model_mapping_warning!)).toBeInTheDocument();
    expect(screen.getByText("Routing Insights")).toBeInTheDocument();
    expect(screen.getByText("0.913")).toBeInTheDocument();
  });

  it("disables submission while the API request is pending", async () => {
    let finish!: (value: ChatResponse) => void;
    vi.spyOn(api, "chat").mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const user = userEvent.setup();
    renderAt("/playground");
    await user.type(screen.getByLabelText("Prompt"), "A prompt");
    await user.click(screen.getByRole("button", { name: /Route & Run/ }));
    expect(screen.getByRole("button", { name: /Routing/ })).toBeDisabled();
    finish(chatResponse);
    expect(await screen.findByText("Request complete")).toBeInTheDocument();
  });

  it("shows API errors on Playground", async () => {
    vi.spyOn(api, "chat").mockRejectedValue(new ApiError("Telemetry unavailable", 503));
    const user = userEvent.setup();
    renderAt("/playground");
    await user.type(screen.getByLabelText("Prompt"), "A prompt");
    await user.click(screen.getByRole("button", { name: /Route & Run/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Telemetry unavailable");
  });

  it("renders real analytics values and model breakdown", async () => {
    vi.spyOn(api, "summary").mockResolvedValue({ ...summary, fallback_rate: 0.5 });
    vi.spyOn(api, "models").mockResolvedValue(models);
    vi.spyOn(api, "recent").mockResolvedValue([record]);
    renderAt("/analytics");
    expect(await screen.findByText("4", { selector: ".kpi-value" })).toBeInTheDocument();
    expect(screen.getByText("25.0%")).toBeInTheDocument();
    expect(screen.getByText("50.0%")).toBeInTheDocument();
    expect(screen.getAllByText("Gemini 2.5 Flash").length).toBeGreaterThan(1);
    expect(screen.getByText("$0.004800")).toBeInTheDocument();
    expect(screen.queryByText("req-test-123")).not.toBeInTheDocument();
  });

  it("renders Settings with live routing cost multipliers and inference_requests schema", async () => {
    vi.spyOn(api, "routingConfig").mockResolvedValue({
      policy_id: "V2.1 Cost-Aware Router",
      quality_floor: 0.9,
      fallback_model: "gemini-2.5-pro",
      provider_mode: "mock",
      telemetry_table: "inference_requests",
      models: [
        {
          canonical_model: "deepseek-v3-0324",
          display_name: "DeepSeek V3",
          provider: "deepseek",
          training_average_cost_usd: 0.000586,
          relative_cost_vs_cheapest: 1.0,
          benchmark_input_price_per_million: 0.25,
          benchmark_output_price_per_million: 0.88,
          provider_model: "deepseek-v3-0324",
        },
        {
          canonical_model: "gemini-2.5-flash",
          display_name: "Gemini 2.5 Flash",
          provider: "gemini",
          training_average_cost_usd: 0.00361,
          relative_cost_vs_cheapest: 6.16,
          benchmark_input_price_per_million: 0.3,
          benchmark_output_price_per_million: 2.5,
          provider_model: "gemini-2.5-flash",
        },
        {
          canonical_model: "gemini-2.5-pro",
          display_name: "Gemini 2.5 Pro",
          provider: "gemini",
          training_average_cost_usd: 0.038847,
          relative_cost_vs_cheapest: 66.3,
          benchmark_input_price_per_million: 1.25,
          benchmark_output_price_per_million: 10.0,
          provider_model: "gemini-2.5-pro",
        },
      ],
    });
    renderAt("/settings");
    expect(await screen.findByText("1.0x (Baseline)")).toBeInTheDocument();
    expect(screen.getByText("~6.2x")).toBeInTheDocument();
    expect(screen.getByText("~66.3x")).toBeInTheDocument();
    expect(screen.getByText(/inference_requests table/)).toBeInTheDocument();
  });

  it("shows an honest empty analytics state", async () => {
    vi.spyOn(api, "summary").mockResolvedValue({ total_requests: 0, average_cost_usd: null, average_latency_ms: null, error_rate: 0 });
    vi.spyOn(api, "models").mockResolvedValue([]);
    vi.spyOn(api, "recent").mockResolvedValue([]);
    renderAt("/analytics");
    expect(await screen.findByText("No inference data yet")).toBeInTheDocument();
    expect(screen.getByText("Run a request in Playground to start building your telemetry.")).toBeInTheDocument();
  });

  it("renders request metadata and a privacy-safe detail drawer", async () => {
    vi.spyOn(api, "requests").mockResolvedValue({ requests: [record], total: 1, limit: 10, offset: 0 });
    const user = userEvent.setup();
    renderAt("/requests");
    expect(await screen.findByText("req-test-123")).toBeInTheDocument();
    expect(screen.getByText("gemini-3.8-flash")).toBeInTheDocument();
    await user.click(screen.getByRole("row", { name: /View request req-test-123/ }));
    expect(screen.getByRole("dialog", { name: "Request details" })).toBeInTheDocument();
    expect(screen.getByText(record.prompt_hash)).toBeInTheDocument();
    expect(screen.getByText("Prompt and response not stored")).toBeInTheDocument();
    expect(screen.queryByText(/Explain binary search/)).not.toBeInTheDocument();
  });

  it("enforces the UI floor and runs the built-in qualitative OOD set", async () => {
    vi.spyOn(api, "chat").mockResolvedValue(chatResponse);
    const user = userEvent.setup();
    renderAt("/playground");
    expect(screen.getByLabelText("Quality target")).toHaveAttribute("min", "0.85");
    expect(screen.getByText("V5 Platt Cost-Aware Router policy floor: 0.85")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: /Evaluation Mode/ }));
    expect(screen.getByRole("heading", { name: "Unseen/OOD Demo Prompts" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Run all prompts/ }));
    expect(await screen.findByText("Complex research and architecture")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(16);
    expect(api.chat).toHaveBeenCalledTimes(15);
    expect(api.explain).toHaveBeenCalledTimes(15);
    expect(screen.getByText(/not part of the benchmark evaluation/i)).toBeInTheDocument();
  });

  it("adapts the Playground slider floor when routingConfig exposes V2.1 (0.90)", async () => {
    vi.spyOn(api, "routingConfig").mockResolvedValue({
      policy_id: "V2.1 Cost-Aware Router",
      quality_floor: 0.9,
      fallback_model: "gemini-2.5-pro",
      provider_mode: "mock",
      telemetry_table: "inference_requests",
      models: [],
    });
    vi.spyOn(api, "explain").mockResolvedValue({ ...routeExplanation, quality_floor: 0.9 });
    vi.spyOn(api, "chat").mockResolvedValue(chatResponse);
    const user = userEvent.setup();
    renderAt("/playground");
    expect(await screen.findByText("V2.1 Cost-Aware Router policy floor: 0.90")).toBeInTheDocument();
    expect(screen.getByLabelText("Quality target")).toHaveAttribute("min", "0.90");
    await user.type(screen.getByLabelText("Prompt"), "What is 2 + 2?");
    await user.click(screen.getByRole("button", { name: /Route & Run/ }));
    expect(api.chat).toHaveBeenCalledWith({ prompt: "What is 2 + 2?", min_quality: 0.9 });
  });
});
