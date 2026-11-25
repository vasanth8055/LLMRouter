# LLMRouter V5 Baseline Report (Phase 0)

- **Timestamp:** `2026-11-25T11:24:05+05:30`
- **Baseline Branch & Commit:** `baseline` at [`173420e`](file:///c:/Users/HP/Project/LLMRouter) (branched from `main` [`d3c9324`](file:///c:/Users/HP/Project/LLMRouter))
- **Active Working Branch:** `improve/v5`
- **Python Environment:** `.\.venv\Scripts\python.exe`

---

## 1. Git Baseline Snapshot

Committed all 48 previously uncommitted / untracked workspace files (Phase 4A, Phase 5, and UI updates) to `baseline` (`173420e`) while keeping `.env` ignored and untracked, then branched `improve/v5`.

---

## 2. Automated Test & Build Baseline

| Check | Command | Result | Exit Code | Notes |
|---|---|---|---:|---|
| **Backend tests** | `.\.venv\Scripts\python.exe -m pytest -q` | **146 passed** in `21.49s` | `0` | 1 `StarletteDeprecationWarning` (`httpx` with `starlette.testclient`); provider tests use `httpx.MockTransport`. |
| **Frontend tests** | `cd frontend; npm.cmd test -- --run` | **8 passed** (1 file) in `2.61s` | `0` | Vitest `v5.0.3`; API client calls are mocked via `vi.spyOn`. |
| **Frontend build** | `cd frontend; npm.cmd run build` | **Built** in `3.34s` | `0` | `tsc -b && vite build` (`v6.4.3`, `2011` modules; JS bundle `481.10 kB`, CSS `34.51 kB`). |

---

## 3. Recorded Ground-Truth State Entering `improve/v5`

- **Frozen runtime model:** `models/router_v21_tfidf.joblib` ($\tau = 0.90$, TF-IDF `(1,2)` `min_df=2, max_features=30000, sublinear_tf=True` + one-hot model + Logistic Regression `max_iter=2000, random_state=42`, fallback `gemini-2.5-pro`).
- **Burned MMLU-Pro `test_3000` held-out split (`N=450`):** `data/router_v21_final_test_results.csv` records V2.1 at `83.78%` vs Always Pro `83.33%` (`10.36%` savings, `63.33%` fallback, McNemar $p = 0.6875$). **Quarantined / burned — will not be used for any new tuning or selection.**
- **Known defects & gaps to address in Phases 1–6:**
  1. Phase 5 5-fold CV (`results/experiments/phase5_cv_metrics.json`) evaluated static `router_v21_tfidf.joblib` without refitting per fold (`70%` in-sample overlap).
  2. No HTTP retry/backoff or runtime provider failover; no application `Dockerfile`.
  3. `frontend/src/pages/Settings.tsx` hardcodes wrong cost multipliers (`~1.4x` / `~9.8x` vs actual training cost ratios `~6.16x` / `~66.30x`) and cites `"requests table"` instead of `inference_requests`.
  4. `frontend/src/pages/Analytics.tsx` hardcodes the `"Fallback rate"` KPI to `"Not available"`.
  5. Live provider generation is **UNVERIFIED** against real APIs.
  6. Cross-domain transfer to LiveCodeBench (`N=1,055`) fell back 99.15% (`100%` Pro routed, `0.00%` savings).
  7. High fallback rate (`63.33%` on `test_3000` test, `53.77%` in Phase 5 CV) limits V2.1 cost savings to `~10.3%`.

---

## 4. Phase 0 Gate Result

- **Gate Status:** **PASS** (all Phase 0 criteria in `reports/pregistered_gates.md` met).
