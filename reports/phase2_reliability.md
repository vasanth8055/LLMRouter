# Phase 2 — Reliability Engineering Report

**Branch:** `improve/v5`  
**Pre-Registered Gate:** [reports/pregistered_gates.md](file:///c:/Users/HP/Project/LLMRouter/reports/pregistered_gates.md) (`2026-11-25T11:37:00+05:30`)

## Summary of Changes

1. **Provider Retries (`429`, `5xx`, Timeouts Only)**
   - Added `llm_max_retries=2`, `llm_retry_base_delay_s=0.25`, `llm_retry_max_delay_s=2.0`, and overall per-request `llm_request_deadline_s=120.0` to `app/config.py`.
   - Extended `ProviderError` (`app/providers/errors.py`) with `retryable: bool = False` and `status_code: int | None = None`; `ProviderRateLimitError` (`429`) and `ProviderTimeoutError` default to `retryable=True`.
   - Updated `raise_for_provider_status` (`app/providers/http_utils.py`) to mark HTTP `>= 500` as `retryable=True` and non-429 `4xx` as non-retryable, and added `async_with_retries` using exponential backoff + jitter (`min(max_delay_s, base_delay_s * 2**attempt) * (0.5 + 0.5 * rand_fn())`) plus context-scoped deadline & retry-count tracking (`REQUEST_DEADLINE_AT`, `REQUEST_RETRY_COUNT`).
   - Wired `DeepSeekProvider` (`app/providers/deepseek.py`) and `GeminiProvider` (`app/providers/gemini.py`) through `async_with_retries` while keeping per-request `timeout=self.timeout_s`.

2. **Runtime Failover & Additive Schema / Idempotent Migration**
   - Updated `RouterService` (`app/services/router_service.py`) to enforce the overall `llm_request_deadline_s` (`120.0s` default) across retries and failover, iterate remaining eligible models (ascending training cost) followed by non-latency-excluded `gemini-2.5-pro`, and record `attempted_models`, `failover`, `policy_fallback`, and `retry_count` in both telemetry and `RouteResponse` (`app/schemas.py`).
   - Added optional fields with safe defaults (`attempted_models`, `failover`, `policy_fallback`, `retry_count`) to `app/schemas.py`, `app/telemetry/models.py`, and `app/telemetry/api_schemas.py` without renaming or removing any existing field.
   - Added idempotent `ALTER TABLE inference_requests ADD COLUMN IF NOT EXISTS ...` statements (4 statements including `retry_count`) in `PostgresTelemetryRepository.migrate()` (`app/telemetry/postgres.py`), called from both `TelemetryService.initialize()` (`app/telemetry/service.py`) and `scripts/init_telemetry_db.py`.
   - Added `fallback_rate` and `failover_rate` to `PostgresTelemetryRepository.analytics_summary()`, and added `GET /v1/routing/config` (`app/main.py`) exposing live pricing/multiplier metadata from `app/providers/model_registry.py` (`$0.25/$0.88`, `$0.30/$2.50`, `$1.25/$10.00` per 1M tokens; training average ratios `1.0x`, `6.16x`, `66.30x`).
   - Applied `response_model_exclude_unset=True` on `/v1/analytics/summary`, `/v1/analytics/recent`, and `/v1/requests` so legacy mocks and old rows preserve exact compatibility.

3. **Frontend Fixes (`Analytics.tsx`, `Settings.tsx`)**
   - Updated `frontend/src/types/api.ts` and `frontend/src/api/client.ts` with `RoutingConfigResponse`, `fallback_rate`, `failover_rate`, `attempted_models`, `failover`, and `policy_fallback`.
   - Replaced hardcoded `"Not available"` fallback KPI in `frontend/src/pages/Analytics.tsx` with live `summary.fallback_rate` (`N/A` if null/absent).
   - Replaced wrong hardcoded multipliers (`~1.4x` / `~9.8x`) and `"requests table"` copy in `frontend/src/pages/Settings.tsx` with live `/v1/routing/config` values (fallback `1.0x / ~6.2x / ~66.3x` and `inference_requests table`).

## Verification & Commands Run

- **Backend test suite** (`tests/test_v5_phase2_reliability.py` + all baseline/Phase 1 tests):
  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q
  ```
  Result: **`163 passed, 1 warning in 26.92s`** (`EXIT=0`) — all 150 pre-existing tests plus 13 Phase 2 reliability tests (including 120s deadline abort and `retry_count` telemetry) passed.

- **Frontend unit/integration tests** (`frontend/src/test/App.test.tsx`):
  ```powershell
  cd frontend; npm.cmd test -- --run
  ```
  Result: **`9 passed (9)`** (`EXIT=0`) — all 8 pre-existing tests plus the new `Settings.tsx` live routing config test and updated `Analytics.tsx` `fallback_rate` test passed.

- **Frontend TypeScript + Vite production build**:
  ```powershell
  cd frontend; npm.cmd run build
  ```
  Result: **`✓ built in 3.34s`** (`EXIT=0`, `dist/assets/index-DBNxtpTe.js 481.71 kB`).

## Gate Assessment

- Criterion 1 (Retries on `429`/`5xx`/timeouts, zero retries on non-429 `4xx`, overall 120s deadline): **PASS** (verified in `tests/test_v5_phase2_reliability.py`).
- Criterion 2 (Runtime failover + additive `RouteResponse` + `retry_count` + idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migration): **PASS** (verified in `tests/test_v5_phase2_reliability.py`).
- Criterion 3 (`150` backend + `8` frontend tests unchanged + new Phase 2 tests): **PASS** (`163 passed` pytest, `9 passed` vitest).
- Criterion 4: **Docker packaging removed by decision, not part of project scope** (`compose.yaml` restored to `postgres` service only; `Dockerfile` and `.dockerignore` deleted).
- Criterion 5 (`npm.cmd run build` passes cleanly): **PASS** (`built in 3.34s`).
