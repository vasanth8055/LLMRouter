# Phase 3 — Live Validation Report

**Status:** **`BLOCKED (no API keys in .env)`** *(not failed)*  
**Branch:** `improve/v5`  
**Probe Output:** [results/v5/live_probe.json](file:///c:/Users/HP/Project/LLMRouter/results/v5/live_probe.json)  
**Pre-Registered Gate:** [reports/pregistered_gates.md](file:///c:/Users/HP/Project/LLMRouter/reports/pregistered_gates.md)

## Status & Deferral Scope

- Live execution is currently **`BLOCKED (no API keys in .env)`** and is **deferred until after Phase 5** so that the eventual live run exercises the final Phase 5 router rather than the frozen V2.1 artifact twice.
- When resumed after Phase 5 with active provider keys, the run will be explicitly labeled **`"pipeline validation"`** — because live provider API model versions may differ from the benchmarked `deepseek-v3-0324`, `gemini-2.5-flash`, and `gemini-2.5-pro` snapshots, **no routing-quality claims will be made from it**.

## `.env` Key & Model Inventory (Names Only)

Inspected via `.\.venv\Scripts\python.exe scripts/live_validation.py --mode probe`:

| Environment Variable | Status | Notes |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | `EMPTY_OR_PLACEHOLDER` | DeepSeek adapter unavailable (`missing DEEPSEEK_API_KEY`) |
| `GEMINI_API_KEY` | `EMPTY_OR_PLACEHOLDER` | Gemini adapters unavailable (`missing GEMINI_API_KEY`) |
| `OPENAI_API_KEY` | `EMPTY_OR_PLACEHOLDER` | Unused by router |
| `AWS_REGION` | `EMPTY_OR_PLACEHOLDER` | Unused by router |
| `DEEPSEEK_API_MODEL` | `EMPTY_OR_PLACEHOLDER` | Required for DeepSeek live mode |
| `GEMINI_FLASH_API_MODEL` | `SET` (`gemini-3.8-flash`) | Differs from canonical `gemini-2.5-flash` |
| `GEMINI_PRO_API_MODEL` | `EMPTY_OR_PLACEHOLDER` | Required for Gemini Pro live mode |
| `LIVE_MODEL_PRICING` | `SET` (`{}`) | Unpopulated; billed cost would fall back to unset unless configured |

- **Live-callable canonical models:** `0` (`[]`).

## Non-Test Prompt Sampling, Cost Estimates & Mocked Failover Check

- **Split Hygiene:** Sampled strictly from the `2,550` non-test MMLU-Pro `test_3000` prompts (`train` + `validation` splits only; `0` overlap with the `450` burned `test_question_ids`).
- **Mocked DeepSeek Failover Smoke Check:** Executed via `run_mocked_deepseek_failover_check` (simulating a `429 ProviderRateLimitError` on `deepseek-v3-0324` → failing over to `gemini-2.5-flash` with `failover=True`, `retry_count=0`). Not a live network call.
- **Estimated Token Cost (Benchmark Rates: `$0.25/$0.88`, `$0.30/$2.50`, `$1.25/$10.00` per 1M tokens):**
  - **`N = 60` (`train`/`validation` stratified sample):** V2.1 routed estimate **`$1.948436`** (`2` DeepSeek, `6` Flash, `52` Pro; Always-DeepSeek `$0.032799`, Always-Flash `$0.304224`, Always-Pro `$2.119057`).
  - **`N = 150` (`train`/`validation` stratified sample):** V2.1 routed estimate **`$4.984235`** (`10` DeepSeek, `12` Flash, `128` Pro; Always-DeepSeek `$0.085429`, Always-Flash `$0.339924`, Always-Pro `$5.594934`).
- **Live Latency / Billed Cost / Offline Cost Gap / Error Rate:** **UNVERIFIED (BLOCKED — no API keys in `.env`)**.
