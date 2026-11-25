# Phase 5 (Partial: Attempts a & b) — Calibration, Embeddings & Latency

- **Generated:** `2026-11-25T06:53:57.302707+00:00`
- **Comparator (`Refit V2.1 (inner-CV tau)`):** fallback `50.77%`, savings `21.59%`, retention `98.44%`, NI LB `-1.93 pp`.
- **Latency Gate:** routing-decision `p95 <= 25.0 ms`.

## Attempt Summary (`results/v5/attempt_log.csv`)

| ID | Method | Acc | Retention | Savings | Fallback | NI LB | ECE (before→after) | Brier (before→after) | Routing p50/p95/p99 (ms) | Cold Start (ms) | Q/C Gate | Lat Gate | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comparator | Refit V2.1 (inner-CV tau) | 84.33% | 98.44% | 21.59% | 50.77% | -1.93 pp | 0.01769 | 0.1461 | 2.17/2.73/3.40 | 108.5 | False | True | False |
| 5a-platt | Attempt 5a-1: Platt Calibration (inner-val fit) | 84.53% | 98.68% | 23.67% | 44.57% | -1.70 pp | 0.01769→0.00864 | 0.1461→0.14626 | 2.55/4.65/5.47 | 108.5 | True | True | True |
| 5a-isotonic | Attempt 5a-2: Isotonic Calibration (inner-val fit) | 84.70% | 98.87% | 19.79% | 43.70% | -1.50 pp | 0.01769→0.02863 | 0.1461→0.15091 | 2.53/4.54/5.54 | 108.5 | False | True | False |
| 5b-minilm | Attempt 5b-1: all-MiniLM-L6-v2 + LR | 84.60% | 98.75% | 14.76% | 65.10% | -1.57 pp | 0.01769→0.03482 | 0.1461→0.15021 | 20.68/28.44/29.60 | 32411.0 | False | False | False |
| 5b-minilm-tfidf | Attempt 5b-2: all-MiniLM-L6-v2 + TF-IDF + LR | 84.13% | 98.21% | 17.14% | 61.90% | -2.10 pp | 0.01769→0.04165 | 0.1461→0.14931 | 21.11/29.37/30.32 | 32411.0 | False | False | False |

## Concurrency Benchmark (Frozen V2.1 `/v1/chat`, Mock Providers)

- `N = 50` requests at concurrency `10`: wall `0.170s`, throughput `294.7 req/s`, `p50 = 18.29 ms`, `p95 = 32.60 ms`, `p99 = 35.62 ms`.

## Paired Bootstrap (`5a-platt` minus `Refit V2.1 comparator`, `10,000` Resamples, Same 5 Outer Folds)

| Metric Difference (`5a-platt - comparator`) | Point Difference | 95% Paired Bootstrap CI | Excludes Zero? |
| --- | --- | --- | --- |
| Accuracy (`pp`) | `+0.20 pp` (`84.53%` vs `84.33%`) | `[-0.37 pp, +0.73 pp]` | **No** (accuracy parity) |
| Cost savings vs Always-Pro (`pp`) | `+2.08 pp` (`23.67%` vs `21.59%`) | `[+0.31 pp, +3.82 pp]` | **Yes** |
| Fallback rate (`pp`) | `-6.20 pp` (`44.57%` vs `50.77%`) | `[-7.80 pp, -4.57 pp]` | **Yes** |

## Honest Caveats & Interpretation

- **Why Platt helped (`threshold/fallback behavior`, NOT calibration error reduction):** Baseline uncalibrated 10-bin ECE was already low (`0.01769`, ~`0.0177`), and Brier score did **not** improve after Platt (`0.14610 → 0.14626`). Although ECE dropped numerically (`0.01769 → 0.00864`), Platt scaling shifted per-model logit slopes/intercepts on `inner_val`, altering which prompts fell just above/below the discrete `0.05`-step threshold grid (`tau*`) — meaning Platt's gain comes from **threshold/fallback behavior** rather than fixing badly miscalibrated probabilities.
- **Multiple-comparison / selection effect:** Five variants (`comparator`, `5a-platt`, `5a-isotonic`, `5b-minilm`, `5b-minilm-tfidf`) were evaluated on the same 5 outer CV folds (`N = 3,000`), creating a post-hoc selection effect when picking the winning variant across candidates.
