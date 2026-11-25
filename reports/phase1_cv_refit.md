# Phase 1 — Leakage-Free 5-Fold Nested CV (Refit V2.1 Recipe)

- **Generated (UTC):** `2026-11-25T05:57:13.862026+00:00`
- **Dataset:** `data\llmrouter_mmlupro_test3000_long.csv` (`N = 3,000` prompts, `9,000` rows)
- **Protocol:** 5-fold `GroupKFold` by `question_id` (`2,400` outer-train / `600` outer-test); inner 80/20 `GroupShuffleSplit` (`1,920` / `480`) selects $\tau \in [0.50, 0.95]$ (step `0.05`) using the $\ge 98\%$ Always-Pro retention rule; full outer-train refit before out-of-fold evaluation.
- **Non-Inferiority Margin:** `-2.00 pp` (95% bootstrap CI lower bound of `router - Always Pro` must exceed `-2.00 pp`).

## 1. Gate Evaluation

- **Zero Prompt Leakage Verified:** `True`
- **Refit V2.1 (inner-CV tau) Quality Retention $\ge 98.0\%$:** `True` (`98.44%`)
- **Refit V2.1 (inner-CV tau) Non-Inferiority ($> -2.00\text{ pp}$):** `True` (lower bound `-1.93 pp`)
- **Phase 1 Gate Passed:** **`PASS`**

## 2. Pooled Out-of-Fold Results (`N = 3,000` Prompts)

| Method | Accuracy (95% CI) | % of Always-Pro | Cost Saving vs Pro (95% CI) | Mean Cost | Fallback Rate | McNemar vs Pro ($p$) | McNemar vs Flash ($p$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always DeepSeek | 78.00% [76.50%, 79.47%] | 91.05% | 98.50% [98.46%, 98.54%] | $0.000587 | N/A | 5.021e-27 | 1.093e-04 |
| Always Gemini Flash | 80.93% [79.50%, 82.33%] | 94.47% | 90.65% [89.59%, 91.60%] | $0.003653 | N/A | 2.117e-14 | 1.000e+00 |
| Always Gemini Pro | 85.67% [84.37%, 86.90%] | 100.00% | 0.00% [0.00%, 0.00%] | $0.039062 | N/A | 1.000e+00 | 2.117e-14 |
| Random (3-model, seed 42) | 81.77% [80.40%, 83.13%] | 95.45% | 63.11% [60.76%, 65.41%] | $0.014411 | N/A | 6.573e-13 | 1.579e-01 |
| Refit V2.1 (inner-CV tau) | 84.33% [83.03%, 85.60%] | 98.44% | 21.59% [19.66%, 23.61%] | $0.030626 | 50.77% | 8.581e-06 | 3.775e-08 |
| Refit V2.1 (fixed tau=0.90) | 85.37% [84.07%, 86.60%] | 99.65% | 9.83% [8.48%, 11.22%] | $0.035224 | 64.40% | 1.221e-01 | 5.255e-13 |
| Frozen V2.1 (in-sample, do not cite) | 85.63% [84.40%, 86.87%] | 99.96% | 10.31% [8.97%, 11.75%] | $0.035033 | 53.77% | 1.000e+00 | 2.495e-14 |

> **Warning on `Frozen V2.1 (in-sample, do not cite)`:** This column evaluates the static `models/router_v21_tfidf.joblib` artifact across all 3,000 prompts; **2,100 of 3,000 (70%) were in its training split**. It is included solely to expose the in-sample inflation identified in the baseline audit.

## 3. Non-Inferiority Analysis vs Always Gemini Pro (Margin = `-2.00 pp`)

| Method | Diff vs Pro (pp) | 95% Bootstrap CI (pp) | CI Lower Bound (pp) | Margin (`-2.00 pp`) | Non-Inferiority |
|---|---:|---:|---:|---:|:---:|
| Refit V2.1 (inner-CV tau) | -1.33 pp | [-1.93, -0.73] pp | -1.93 pp | -2.00 pp | **PASS** |
| Refit V2.1 (fixed tau=0.90) | -0.30 pp | [-0.63, +0.03] pp | -0.63 pp | -2.00 pp | **PASS** |
| Always Gemini Flash | -4.73 pp | [-5.93, -3.50] pp | -5.93 pp | -2.00 pp | **FAIL** |
| Always DeepSeek | -7.67 pp | [-9.07, -6.30] pp | -9.07 pp | -2.00 pp | **FAIL** |
| Random (3-model, seed 42) | -3.90 pp | [-5.00, -2.87] pp | -5.00 pp | -2.00 pp | **FAIL** |
| Frozen V2.1 (in-sample, do not cite) | -0.03 pp | [-0.23, +0.17] pp | -0.23 pp | -2.00 pp | **PASS** |

## 4. Per-Fold Inner-CV Threshold Selection & Routing Distribution

| Outer Fold | Inner Train / Val | Outer Test | Selected $\tau^*$ | Inner Val Acc | Inner Val Retention | Inner Val Savings |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1920 / 480 | 600 | 0.85 | 83.12% | 98.76% | 34.52% |
| 2 | 1920 / 480 | 600 | 0.90 | 85.62% | 99.76% | 9.22% |
| 3 | 1920 / 480 | 600 | 0.90 | 85.42% | 99.27% | 7.69% |
| 4 | 1920 / 480 | 600 | 0.85 | 83.54% | 98.28% | 33.97% |
| 5 | 1920 / 480 | 600 | 0.90 | 86.25% | 99.76% | 6.96% |

| Method | DeepSeek (%) | Flash (%) | Pro (%) | Fallbacks |
|---|---:|---:|---:|---:|
| Refit V2.1 (inner-CV tau) | 14.60% (438) | 10.47% (314) | 74.93% (2248) | 1523 (50.77%) |
| Refit V2.1 (fixed tau=0.90) | 6.63% (199) | 5.63% (169) | 87.73% (2632) | 1932 (64.40%) |
| Frozen V2.1 (in-sample, do not cite) | 7.53% (226) | 5.80% (174) | 86.67% (2600) | 1613 (53.77%) |

## 5. Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/run_v5_phase1_cv_refit.py
```
