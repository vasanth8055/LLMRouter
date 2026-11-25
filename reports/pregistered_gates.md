# Pre-Registered Experiment & Engineering Gates (`improve/v5`)

> Gates are written and committed **before** executing each phase and are never modified retroactively.

---

## Phase 0 — Baseline Safety Gate
- **Timestamp:** `2026-11-25T11:23:30+05:30`
- **Branch:** `improve/v5` (branched from `baseline` commit `173420e`)
- **Pass Criteria:**
  1. Uncommitted working tree safely committed to `baseline` without `.env` or secrets (`git status` clean).
  2. `.\.venv\Scripts\python.exe -m pytest -q` passes all existing backend tests (`>= 146` passed, `0` failed, `0` skipped).
  3. `npm.cmd test -- --run` passes all existing frontend tests (`>= 8` passed, `0` failed).
  4. `npm.cmd run build` (`tsc -b && vite build`) exits `0` with no TypeScript or bundle errors.
  5. Baseline recorded in `reports/v5_baseline.md`.

---

## Phase 1 — Leakage-Free 5-Fold Nested CV (Refit V2.1 Recipe)
- **Timestamp:** `2026-11-25T11:26:00+05:30`
- **Dataset:** `data/llmrouter_mmlupro_test3000_long.csv` (`N = 3,000` prompts, `9,000` rows; 3 models per prompt). The burned 450-prompt test split is NOT treated as a separate tuning/selection set; all 3,000 prompts are partitioned by `GroupKFold(n_splits=5)` grouped by `question_id` (`2,400` outer-train / `600` outer-test per fold).
- **Refit Protocol:** Inside each outer fold:
  1. Split outer-train (`2,400` questions) into inner-train (`1,920` questions, 80%) and inner-val (`480` questions, 20%) via `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42 + fold)`.
  2. Fit TF-IDF (`ngram_range=(1, 2)`, `min_df=2`, `max_features=30000`, `sublinear_tf=True`) + OneHotEncoder(`model`) + `LogisticRegression(C=1.0, max_iter=2000, random_state=42)` on inner-train only.
  3. Sweep `tau` over the exact V2.1 grid `[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]` on inner-val using inner-train average model costs (`deepseek-v3-0324`, `gemini-2.5-flash`, `gemini-2.5-pro`) and `gemini-2.5-pro` fallback; select `tau*` maximizing cost savings vs Always-Pro subject to `inner_val_acc >= 0.98 * inner_val_pro_acc` (if none qualify, select highest accuracy, tie-break highest `tau`).
  4. Refit the exact V2.1 predictor pipeline and recompute training-average model costs on the full `2,400` outer-train questions, then evaluate once out-of-fold on the `600` outer-test questions using `tau*`.
  5. Also evaluate `Refit V2.1 (fixed tau=0.90)` out-of-fold per outer fold as a threshold-isolation diagnostic, and include `Frozen V2.1 (in-sample, do not cite)` strictly labeled as contaminated by 70% training overlap.
- **Pre-Registered Pass Criteria (Gate):**
  1. **Zero leakage:** Outer test `question_id`s have `0` overlap with inner-train, inner-val, and outer-train in every fold (`5/5` folds).
  2. **Non-inferiority vs Always-Pro:** Lower bound of the 95% paired bootstrap CI (`10,000` resamples, seed `42`) for `(Refit V2.1 accuracy - Always-Pro accuracy)` is **`> -2.00 percentage points`** (`-0.02`).
  3. **Quality retention:** Pooled out-of-fold accuracy of `Refit V2.1 (inner-CV tau)` is **`>= 98.0%`** of Always-Pro accuracy.
  4. **Outputs & tests:** `results/v5/cv_refit_v21.json`, `reports/phase1_cv_refit.md`, and automated tests verifying the nested CV protocol pass. *(Note: if Refit V2.1 fails criteria 2 or 3, that negative/null result is reported honestly as a valid Phase 1 deliverable.)*

---

## Phase 5 — Attack the Fallback Rate (Pre-Registered Comparator Gate)
- **Timestamp:** `2026-11-25T11:35:45+05:30` (Routing-latency requirement added `2026-11-25T12:00:00+05:30` before Phase 4)
- **Comparator:** `"Refit V2.1 (inner-CV tau)"` from Phase 1 (`results/v5/cv_refit_v21.json`: fallback `50.77%`, savings `21.59%`, retention `98.44%`, NI lower bound `-1.93 pp`).
- **Pre-Registered Pass Criteria:**
  1. **Routing & Accuracy Criterion:** A Phase 5 candidate is **better** only if, using the **same 5 outer folds (`N = 3,000`, `GroupKFold(question_id)`)** with inner-CV tuned thresholds, it beats `"Refit V2.1 (inner-CV tau)"` on **cost savings (`> 21.59%`) AND fallback rate (`< 50.77%`)** while keeping **quality retention `>= 98.0%`** and **non-inferiority 95% CI lower bound `> -2.00 pp`**. Otherwise it is **NOT better**.
  2. **Routing-Latency & Concurrency Criterion:** Measure and report routing-decision latency (`p50` / `p95` / `p99`), cold-start model load/first-prediction latency, and a concurrency benchmark against `/v1/chat` (with mock providers); Phase 5 candidate **routing-decision `p95` latency must be `<= 25 ms`**.

---

## Phase 2 — Reliability Engineering Gate
- **Timestamp:** `2026-11-25T11:37:00+05:30`
- **Pass Criteria:**
  1. Provider adapters implement configurable exponential backoff + jitter retries (`max_retries=2` default) strictly on HTTP `429`, `5xx`, and timeouts; zero retries on non-429 `4xx`.
  2. `RouterService` implements runtime failover across remaining eligible models (ascending cost) then `gemini-2.5-pro`, recording `attempted_models`, `failover`, and `policy_fallback` in additive `RouteResponse` and PostgreSQL telemetry (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migration).
  3. All existing `150` backend tests and `8` frontend tests continue to pass unchanged, plus new tests covering retry-then-success, no retry on `4xx`, failover chain, all-providers-fail `502`/`503`/`504`, idempotent DB migration, `Settings.tsx` real cost multipliers, and `Analytics.tsx` live `fallback_rate` KPI.
  4. **[Removed: Docker packaging removed by decision, not part of project scope]** *(Original text: `Dockerfile` and `compose.yaml` (`api` + `postgres`) validate via `docker compose config`; `docker build` verified if Docker daemon is running, otherwise explicitly marked `UNVERIFIED`.)*
  5. `npm.cmd run build` (`tsc -b && vite build`) passes cleanly.

---

## Phase 3 — Live Validation Gate
- **Timestamp:** `2026-11-25T11:53:40+05:30` (Status updated `2026-11-25T12:00:00+05:30`)
- **Status:** **`BLOCKED (no API keys in .env)`** — *not failed*. Deferred until after Phase 5 so it runs against the final router. When resumed, it will be labeled **`"pipeline validation"`** (live model versions may differ from benchmarked ones; routing-quality claims will NOT be made from it).
- **Pass Criteria:**
  1. Inventory `.env` key names (names only, never values) and report which canonical models can be called live (`3-model live validation`, `Gemini-only live validation`, or `BLOCKED` if no API keys exist).
  2. Sample prompts strictly from `train`/`validation` MMLU-Pro `test_3000` splits (`0` overlap with the `450` burned `test_question_ids`).
  3. Execute a 3-prompt probe per available model (`--mode probe`), reporting latency, tokens, errors/rate limits, estimated cost for `N=60` and `N=150`, and a mocked-DeepSeek failover check when DeepSeek is unavailable; stop after probe before running full `N=60`/`N=150` live validation.
  4. Full live validation script checkpoints each result immediately to `results/v5/live_validation.jsonl` (prompt hashes + IDs only, no raw prompts), supports resume, throttles under observed probe limits, enforces request-count cap (`60` default, max `150`) + spend approval (`--approve-spend`), and reports latency `p50`/`p95`, tokens, billed cost, offline cost formula gap, accuracy, retries, and failures.

---

## Phase 4 — Cross-Domain Generalization Gate
- **Timestamp:** `2026-11-25T12:01:30+05:30`
- **Datasets (`N = 4,253` unique prompts / `12,759` rows, zero prompt overlap):**
  - `mmlupro`: `data/llmrouter_mmlupro_test3000_long.csv` (`N = 3,000`; `data/llmrouter_mmlupro_long.csv` is excluded because `998/1,000` prompts overlap `test_3000`).
  - `gpqa`: `data/llmrouter_gpqa_long.csv` (`N = 198`).
  - `livecodebench`: `data/llmrouter_livecodebench_long.csv` (`N = 1,055`).
- **Pre-Registered Evaluation Protocol & Criteria:**
  1. **Zero cross-split / cross-dataset leakage:** Unique `group_id = f"{dataset}:{question_id}"`, verified zero prompt overlap across datasets and zero `group_id` overlap across inner/outer splits.
  2. **Feature variants:** Compare `text_only` (TF-IDF + model OHE) vs `with_domain` (adding inference-time prompt-derived domain feature `infer_domain(prompt)`) under both **Leave-One-Dataset-Out (LODO)** and **Pooled In-Domain 5-Fold Nested CV**.
  3. **Metrics & Inference:** Report per-dataset and pooled accuracy, % of Always-Pro retention, % cost savings vs Always-Pro, fallback rate, routing shares, 95% paired bootstrap CIs (`10,000` rounds, seed `42`), non-inferiority lower bound vs `-2.00 pp`, and paired two-sided McNemar `p`-values vs Always-Pro and Always-Flash.
  4. **Explicit LiveCodeBench Diagnostic:** Quantify and explain whether LiveCodeBench still falls back 100% to `gemini-2.5-pro` under V2.1, LODO, and pooled in-domain CV (report predicted `P(correct | prompt, model)` distributions and true model rankings `DeepSeek > Flash`).
  5. **Gate:** Honest reporting of both positive and negative transfer results (cross-domain OOD transfer is NOT assumed to pass `-2.00 pp` NI if negative transfer occurs; all results saved to `results/v5/cross_domain.json` and `reports/phase4_cross_domain.md` with passing tests).



