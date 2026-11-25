# Phase 5 — Final Report (Attempts c, d, Cached MiniLM & Concurrency)

- **Generated:** `2026-11-25T07:07:33.784916+00:00`
- **Pre-Registered Comparator (`Refit V2.1 (inner-CV tau)`):** fallback `50.77%`, savings `21.59%`, retention `98.44%`, NI LB `-1.93 pp`.
- **Final Chosen Router:** **`5a-platt (Platt-calibrated V2.1 TF-IDF + LR)`** (note: `5a-platt` passes all quality/cost and latency gates without requiring subject metadata; benefit is driven by threshold/fallback behavior rather than calibration error reduction, with 6 variants compared on the same 5 outer folds).

## 1. Cached MiniLM Cold Start vs First-Run Download+Load

- **First-run cold start (earlier download + HF init + encode):** `32,410.98 ms` (`32.41 s`).
- **Cached cold start (`local_files_only=True`, fresh Python process):** total `5939.3 ms` (`import = 5591.2 ms`, `cached model load = 310.1 ms`, `first encode = 38.0 ms`).
- **Single-prompt warm CPU inference (from Phase 5a/b):** `p50 = 20.68 ms`, `p95 = 28.44 ms`, `p99 = 29.60 ms` — **still exceeds the `25.0 ms` `p95` gate**, confirming why GBT-on-embeddings was skipped.

## 2. Attempt (c) — Subject + Prompt Metadata (`requires subject label at inference`)

- **Accuracy:** `83.83%` (`[82.53%, 85.13%]`), **Pro Retention:** `97.86%`
- **Cost Savings vs Always-Pro:** `25.55%` (`[23.61%, 27.65%]`), **Fallback Rate:** `48.37%`
- **NI LB vs Pro:** `-2.50 pp` | **McNemar vs Pro:** `4.981e-08` | **McNemar vs Flash:** `3.975e-06`
- **Calibration:** ECE `0.02199`, Brier `0.14598` | **Routing Latency:** `p50=16.51 ms`, `p95=24.73 ms`, `p99=30.39 ms`
- **Quality/Cost Gate:** **`False`** | **Latency Gate:** **`True`** | **Overall Gate:** **`False`**

## 3. Attempt (d) — Cost–Quality Frontier on Platt Router (`0.80–0.95`, 5-Fold Nested CV)

- **Descriptive Frontier & Selection Caveats:** This cost–quality frontier is **descriptive** (for characterizing the floor tradeoff across `[0.800, 0.950]`, not for threshold tuning). Across the Phase 5 attempt log (`results/v5/attempt_log.csv`), only **1 of 6 attempts passed** the pre-registered gates (`5a-platt`; the `comparator`, `5a-isotonic`, `5b-minilm`, `5b-minilm-tfidf`, and `5c-subject-meta` all failed at least one gate).
- **Fixed-`0.90` Comparison vs Phase 1:** At a fixed `tau = 0.90`, the Platt-calibrated router achieves **`6.84%` savings** (`76.00%` fallback, `99.84%` retention), which is **worse** than Phase 1's uncalibrated `Refit V2.1 (fixed tau=0.90)` result (**`9.83%` savings**, `64.40%` fallback, `99.65%` retention) — confirming again that `5a-platt`'s gain (`23.67%` savings, `44.57%` fallback) comes from selecting `tau* = 0.85` on most inner validation folds rather than uniform improvement at every threshold.

Plot saved to [reports/phase5_frontier.png](file:///c:/Users/HP/Project/LLMRouter/reports/phase5_frontier.png).

| Floor (`tau`) | Accuracy [95% CI] | Pro Retention [95% CI] | Savings vs Pro [95% CI] | Fallback Rate [95% CI] | NI LB vs Pro | Share DS/Fl/Pr (%) |
| --- | --- | --- | --- | --- | --- | --- |
| `0.800` | 81.30% [79.87, 82.67] | 94.90% [93.73, 96.04] | 55.41% [52.97, 57.92] | 15.83% [14.53, 17.17] | -5.40 pp | 39.4/24.0/36.6 |
| `0.825` | 83.10% [81.73, 84.43] | 97.00% [96.05, 97.96] | 41.97% [39.60, 44.35] | 25.00% [23.43, 26.57] | -3.37 pp | 23.5/26.1/50.4 |
| `0.850` | 84.07% [82.73, 85.33] | 98.13% [97.37, 98.89] | 27.64% [25.55, 29.80] | 37.60% [35.87, 39.33] | -2.27 pp | 11.1/22.5/66.4 |
| `0.875` | 84.77% [83.47, 86.03] | 98.95% [98.40, 99.49] | 16.46% [14.84, 18.18] | 55.90% [54.13, 57.67] | -1.37 pp | 4.9/15.5/79.6 |
| `0.900` | 85.53% [84.27, 86.77] | 99.84% [99.53, 100.15] | 6.84% [5.77, 8.00] | 76.00% [74.43, 77.50] | -0.40 pp | 1.7/7.2/91.0 |
| `0.925` | 85.70% [84.43, 86.93] | 100.04% [99.92, 100.16] | 2.67% [2.03, 3.36] | 90.93% [89.87, 91.93] | -0.07 pp | 0.5/3.2/96.3 |
| `0.950` | 85.67% [84.40, 86.90] | 100.00% [100.00, 100.00] | 0.79% [0.47, 1.16] | 97.50% [96.93, 98.07] | +0.00 pp | 0.0/0.9/99.1 |

## 4. Final Chosen Router Concurrency Benchmark (In-Process ASGI + MockProvider, `/v1/chat`, 500 Requests/Level)

- **Methodology & Scope:** Run **in-process against the ASGI application** (`httpx.AsyncClient(transport=httpx.ASGITransport(app=app))`) inside a single Python `asyncio` event loop with `MockProvider` and `NoopTelemetryRepository` — **NOT** over TCP/HTTP against a separate multi-worker `uvicorn` server, real provider APIs, or PostgreSQL. All figures below are **in-process ASGI + MockProvider request latencies**, not real-network end-to-end latencies.
- **Queueing Note (`c = 50`):** `predict_qualities` executes synchronous CPU-bound scikit-learn TF-IDF + Logistic Regression inference (`~4–5 ms` per prompt) on the event loop thread. At `c = 50`, `p95 = 234.46 ms` (`p99 = 271.17 ms`) reflects **in-process event-loop queueing delay** across 50 contending tasks (`50 × ~4.7 ms ≈ 235 ms`), **not per-prompt routing computation time** (which stays `~4.65 ms` `p95`). Single-process throughput plateaus at `~180–201 req/s`.

| Concurrency (In-Process ASGI Tasks) | Requests | Wall Time (s) | In-Process Throughput (req/s) | In-Process p50 (ms) | In-Process p95 (ms) | In-Process p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| `1` | 500 | 2.484 | 201.3 | 4.37 | 7.51 | 9.24 |
| `10` | 500 | 2.779 | 179.9 | 29.07 | 60.53 | 77.98 |
| `50` | 500 | 2.521 | 198.4 | 125.17 | 234.46 | 271.17 |
