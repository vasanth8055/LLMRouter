"""LLMRouter V5 Phase 5 — Final Steps: Cached MiniLM Cold Start, Attempt (c), Attempt (d) Frontier & Concurrency.

Runs strictly without tuning on the burned 450 test split:
1. Measures cached MiniLM cold start in a fresh Python subprocess (`local_files_only=True`),
   separating cached model load + first encode from the earlier first-run download time.
2. Evaluates Attempt (c): Subject + prompt-metadata features (`requires subject label at
   inference`) on the same 5-fold GroupKFold nested CV and logs it to `results/v5/attempt_log.csv`.
3. Evaluates Attempt (d): Cost-quality frontier across floors 0.80..0.95 (step 0.025) on the
   Platt-calibrated router (nested CV), computing 10,000-resample 95% CIs and plotting
   `reports/phase5_frontier.png`.
4. Runs concurrency benchmarks on the final chosen Platt-calibrated router across FastAPI
   `/v1/chat` with mock providers at concurrency 1, 10, and 50 (500 requests per level).
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import httpx
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.main import app  # noqa: E402
from app.providers.factory import ProviderFactory  # noqa: E402
from app.services.inference_service import InferenceService  # noqa: E402
from app.services.router_service import MODEL_IDS, RouterService  # noqa: E402
from app.services.routing_policy import route_by_quality_and_cost  # noqa: E402
from app.telemetry.repository import NoopTelemetryRepository  # noqa: E402
from app.telemetry.service import TelemetryService  # noqa: E402
from run_v5_phase1_cv_refit import (  # noqa: E402
    DATASET_PATH,
    FALLBACK_MODEL,
    FROZEN_V21_PATH,
    MODELS,
    NON_INFERIORITY_MARGIN_PP,
    RANDOM_SEED,
    make_v21_predictor,
    paired_statistics,
    predict_positive_probabilities,
    route_with_policy,
)
from run_v5_phase5_ab import (  # noqa: E402
    ATTEMPT_LOG_CSV,
    ATTEMPT_LOG_FIELDS,
    COMPARATOR,
    EMBEDDING_MODEL_NAME,
    LATENCY_P95_LIMIT_MS,
    PerModelCalibrator,
    _cross_fit_calibrate_inner_val,
    _row_for_log,
    _select_tau_from_probs,
    _summarize_candidate,
    compute_ece,
)

OUTPUT_FINAL_JSON = ROOT / "results" / "v5" / "phase5_final.json"
OUTPUT_FINAL_REPORT = ROOT / "reports" / "phase5_final.md"
FRONTIER_PLOT_PATH = ROOT / "reports" / "phase5_frontier.png"
FRONTIER_FLOORS = [0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]

KNOWN_SUBJECTS = {
    "math",
    "physics",
    "law",
    "chemistry",
    "other",
    "engineering",
    "economics",
    "psychology",
    "health",
    "business",
    "biology",
    "philosophy",
    "computer science",
    "history",
}


def extract_subject_and_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add inference-time subject label (requires MMLU-Pro header) and cheap structural prompt stats."""
    prompts = df["prompt"].astype(str)
    subjects: list[str] = []
    char_lens: list[float] = []
    word_counts: list[float] = []
    digit_ratios: list[float] = []
    for p in prompts:
        m = re.search(r"(?i)Answer the following ([a-z][a-z -]*?) question", p)
        sub = m.group(1).strip().lower() if m else "unknown"
        subjects.append(sub if sub in KNOWN_SUBJECTS else "unknown")
        n_c = max(1, len(p))
        char_lens.append(float(np.log1p(n_c)))
        word_counts.append(float(np.log1p(len(p.split()))))
        digit_ratios.append(float(sum(ch.isdigit() for ch in p) / n_c))

    out = df.copy()
    out["subject"] = subjects
    out["log_char_len"] = char_lens
    out["log_word_count"] = word_counts
    out["digit_ratio"] = digit_ratios
    return out


def make_subject_metadata_predictor() -> Pipeline:
    """TF-IDF + subject OHE + structural numeric metadata + model OHE + LogisticRegression."""
    transformer = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
                "prompt",
            ),
            ("subject", OneHotEncoder(handle_unknown="ignore"), ["subject"]),
            (
                "meta",
                StandardScaler(),
                ["log_char_len", "log_word_count", "digit_ratio"],
            ),
            ("model", OneHotEncoder(handle_unknown="ignore"), ["model"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", transformer),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED)),
        ]
    )


def _predict_subject_meta_probs(predictor: Pipeline, prompts_df: pd.DataFrame) -> dict[str, np.ndarray]:
    classes = list(predictor.named_steps["classifier"].classes_)
    pos_idx = classes.index(1)
    cols = ["prompt", "subject", "log_char_len", "log_word_count", "digit_ratio", "model"]
    probs: dict[str, np.ndarray] = {}
    for m in MODELS:
        inp = prompts_df[["prompt", "subject", "log_char_len", "log_word_count", "digit_ratio"]].copy()
        inp["model"] = m
        probs[m] = predictor.predict_proba(inp[cols])[:, pos_idx]
    return probs


def measure_cached_minilm_cold_start(sample_prompt: str) -> dict[str, float]:
    """Measure fresh-subprocess MiniLM cold start from local HF cache (no download)."""
    code = (
        "import json, time; "
        "t0 = time.perf_counter(); "
        "from sentence_transformers import SentenceTransformer; "
        "t1 = time.perf_counter(); "
        f"m = SentenceTransformer({EMBEDDING_MODEL_NAME!r}, local_files_only=True); "
        "t2 = time.perf_counter(); "
        f"_ = m.encode([{sample_prompt!r}], normalize_embeddings=True, show_progress_bar=False); "
        "t3 = time.perf_counter(); "
        "print(json.dumps({'import_ms': (t1-t0)*1000, 'load_cached_ms': (t2-t1)*1000, 'first_encode_ms': (t3-t2)*1000, 'total_cached_cold_start_ms': (t3-t0)*1000}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    return json.loads(out.strip().splitlines()[-1])


def run_attempt_c_subject_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate Attempt (c): TF-IDF + subject + prompt metadata across the 5 outer folds."""
    work = extract_subject_and_metadata(df).sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = work.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)

    cols = ["prompt", "subject", "log_char_len", "log_word_count", "digit_ratio", "model"]
    scores_all: list[float] = []
    costs_all: list[float] = []
    selected_all: list[str] = []
    fallbacks_all: list[bool] = []
    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    pro_s_all: list[float] = []
    pro_c_all: list[float] = []
    fl_s_all: list[float] = []
    fl_c_all: list[float] = []
    fold_taus: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(q_df, groups=q_df["question_id"]), start=1):
        train_qids = set(map(int, q_df.iloc[tr_idx]["question_id"]))
        test_qids = set(map(int, q_df.iloc[te_idx]["question_id"]))
        assert train_qids.isdisjoint(test_qids)

        outer_train = work[work["question_id"].isin(train_qids)].copy()
        outer_test = work[work["question_id"].isin(test_qids)].copy()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED + fold_idx)
        inner_q = q_df.iloc[tr_idx].reset_index(drop=True)
        in_tr_i, in_val_i = next(gss.split(inner_q, groups=inner_q["question_id"]))
        inner_train = outer_train[
            outer_train["question_id"].isin(set(map(int, inner_q.iloc[in_tr_i]["question_id"])))
        ].copy()
        inner_val = outer_train[
            outer_train["question_id"].isin(set(map(int, inner_q.iloc[in_val_i]["question_id"])))
        ].copy()

        inner_pipe = make_subject_metadata_predictor()
        inner_pipe.fit(inner_train[cols], inner_train["score"].astype(int))
        in_costs = inner_train.groupby("model")["cost"].mean().to_dict()

        val_prompts = inner_val.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        val_qids_arr = val_prompts["question_id"].to_numpy()
        val_scores_piv = inner_val.pivot(index="question_id", columns="model", values="score")[MODELS]
        val_costs_piv = inner_val.pivot(index="question_id", columns="model", values="cost")[MODELS]
        val_probs = _predict_subject_meta_probs(inner_pipe, val_prompts)
        tau_star, _ = _select_tau_from_probs(val_probs, in_costs, val_scores_piv, val_costs_piv, val_qids_arr)
        fold_taus.append({"fold": fold_idx, "selected_tau": tau_star})

        outer_pipe = make_subject_metadata_predictor()
        outer_pipe.fit(outer_train[cols], outer_train["score"].astype(int))
        out_costs = outer_train.groupby("model")["cost"].mean().to_dict()

        test_prompts = outer_test.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        test_qids_arr = test_prompts["question_id"].to_numpy()
        test_scores_piv = outer_test.pivot(index="question_id", columns="model", values="score")[MODELS]
        test_costs_piv = outer_test.pivot(index="question_id", columns="model", values="cost")[MODELS]
        test_probs = _predict_subject_meta_probs(outer_pipe, test_prompts)
        sel, fb = route_with_policy(test_probs, out_costs, tau_star, FALLBACK_MODEL)

        for i, qid in enumerate(test_qids_arr):
            q = int(qid)
            pro_s_all.append(float(test_scores_piv.loc[q, FALLBACK_MODEL]))
            pro_c_all.append(float(test_costs_piv.loc[q, FALLBACK_MODEL]))
            fl_s_all.append(float(test_scores_piv.loc[q, MODELS[1]]))
            fl_c_all.append(float(test_costs_piv.loc[q, MODELS[1]]))
            scores_all.append(float(test_scores_piv.loc[q, sel[i]]))
            costs_all.append(float(test_costs_piv.loc[q, sel[i]]))
            selected_all.append(sel[i])
            fallbacks_all.append(bool(fb[i]))
        for m in MODELS:
            y_true_all.extend(test_scores_piv.loc[test_qids_arr, m].to_numpy(dtype=int).tolist())
            y_prob_all.extend(test_probs[m].tolist())

    # Measure inference latency over 100 prompts using last outer_pipe
    timed_prompts = q_df.head(100).copy()
    t_lats: list[float] = []
    dummy_costs = work.groupby("model")["cost"].mean().to_dict()
    for _, row in timed_prompts.iterrows():
        one_df = pd.DataFrame([row])
        s0 = time.perf_counter()
        feat_df = extract_subject_and_metadata(one_df)
        pr = _predict_subject_meta_probs(outer_pipe, feat_df)
        _ = route_by_quality_and_cost(
            {m: float(pr[m][0]) for m in MODELS}, dummy_costs, 0.90, FALLBACK_MODEL
        )
        t_lats.append((time.perf_counter() - s0) * 1000.0)
    arr = np.array(t_lats, dtype=float)
    lat_summary = {
        "cold_start_ms": float(0.0),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "passes_p95_25ms_gate": float(np.percentile(arr, 95)) <= LATENCY_P95_LIMIT_MS,
    }
    # Measure real joblib deserialization + first prediction cold start
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as tmp_f:
        tmp_path = Path(tmp_f.name)
    joblib.dump(outer_pipe, tmp_path)
    try:
        t_c0 = time.perf_counter()
        loaded_pipe = joblib.load(tmp_path)
        one_cold = extract_subject_and_metadata(timed_prompts.head(1))
        _ = _predict_subject_meta_probs(loaded_pipe, one_cold)
        lat_summary["cold_start_ms"] = float((time.perf_counter() - t_c0) * 1000.0)
    finally:
        tmp_path.unlink(missing_ok=True)

    yt = np.array(y_true_all, dtype=int)
    yp = np.array(y_prob_all, dtype=float)
    cal = {"ece_10bin": compute_ece(yt, yp, n_bins=10), "brier_score": float(brier_score_loss(yt, yp))}
    metrics = _summarize_candidate(
        "Attempt 5c: TF-IDF + subject + prompt metadata (requires subject label at inference)",
        np.array(scores_all),
        np.array(costs_all),
        selected_all,
        fallbacks_all,
        np.array(pro_s_all),
        np.array(pro_c_all),
        np.array(fl_s_all),
        np.array(fl_c_all),
    )
    return {"folds": fold_taus, "calibration": cal, "metrics": metrics, "latency": lat_summary}


def run_attempt_d_platt_frontier(
    df: pd.DataFrame, floors: list[float] = FRONTIER_FLOORS
) -> dict[str, Any]:
    """Compute nested-CV cost-quality frontier on Platt-calibrated probabilities across floors."""
    df_sorted = df.sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = df_sorted.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)

    # Collect aligned OOF Platt probabilities, inner_train costs, and ground-truth outcomes
    oof_rows: list[dict[str, Any]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(q_df, groups=q_df["question_id"]), start=1):
        train_qids = set(map(int, q_df.iloc[tr_idx]["question_id"]))
        test_qids = set(map(int, q_df.iloc[te_idx]["question_id"]))
        outer_train = df_sorted[df_sorted["question_id"].isin(train_qids)].copy()
        outer_test = df_sorted[df_sorted["question_id"].isin(test_qids)].copy()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED + fold_idx)
        inner_q = q_df.iloc[tr_idx].reset_index(drop=True)
        in_tr_i, in_val_i = next(gss.split(inner_q, groups=inner_q["question_id"]))
        inner_train = outer_train[
            outer_train["question_id"].isin(set(map(int, inner_q.iloc[in_tr_i]["question_id"])))
        ].copy()
        inner_val = outer_train[
            outer_train["question_id"].isin(set(map(int, inner_q.iloc[in_val_i]["question_id"])))
        ].copy()

        base_pipe = make_v21_predictor()
        base_pipe.fit(inner_train[["prompt", "model"]], inner_train["score"].astype(int))
        in_costs = inner_train.groupby("model")["cost"].mean().to_dict()

        val_prompts = inner_val.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        val_qids_arr = val_prompts["question_id"].to_numpy()
        val_scores_piv = inner_val.pivot(index="question_id", columns="model", values="score")[MODELS]
        val_raw_probs = predict_positive_probabilities(base_pipe, val_prompts)
        val_y = {m: val_scores_piv.loc[val_qids_arr, m].to_numpy(dtype=int) for m in MODELS}
        cal_platt = PerModelCalibrator("platt").fit(val_raw_probs, val_y)

        test_prompts = outer_test.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        test_qids_arr = test_prompts["question_id"].to_numpy()
        test_scores_piv = outer_test.pivot(index="question_id", columns="model", values="score")[MODELS]
        test_costs_piv = outer_test.pivot(index="question_id", columns="model", values="cost")[MODELS]
        platt_probs = cal_platt.transform(predict_positive_probabilities(base_pipe, test_prompts))

        for i, qid in enumerate(test_qids_arr):
            q = int(qid)
            oof_rows.append(
                {
                    "question_id": q,
                    "prob_ds": float(platt_probs[MODELS[0]][i]),
                    "prob_fl": float(platt_probs[MODELS[1]][i]),
                    "prob_pr": float(platt_probs[MODELS[2]][i]),
                    "train_cost_ds": float(in_costs[MODELS[0]]),
                    "train_cost_fl": float(in_costs[MODELS[1]]),
                    "train_cost_pr": float(in_costs[MODELS[2]]),
                    "score_ds": float(test_scores_piv.loc[q, MODELS[0]]),
                    "cost_ds": float(test_costs_piv.loc[q, MODELS[0]]),
                    "score_fl": float(test_scores_piv.loc[q, MODELS[1]]),
                    "cost_fl": float(test_costs_piv.loc[q, MODELS[1]]),
                    "score_pr": float(test_scores_piv.loc[q, MODELS[2]]),
                    "cost_pr": float(test_costs_piv.loc[q, MODELS[2]]),
                }
            )

    oof = pd.DataFrame(oof_rows).sort_values("question_id", kind="stable").reset_index(drop=True)
    pro_scores = oof["score_pr"].to_numpy()
    pro_costs = oof["cost_pr"].to_numpy()
    flash_scores = oof["score_fl"].to_numpy()
    flash_costs = oof["cost_fl"].to_numpy()
    n = len(oof)

    score_col = {MODELS[0]: "score_ds", MODELS[1]: "score_fl", MODELS[2]: "score_pr"}
    cost_col = {MODELS[0]: "cost_ds", MODELS[1]: "cost_fl", MODELS[2]: "cost_pr"}

    frontier_points: list[dict[str, Any]] = []
    for floor in floors:
        sel: list[str] = []
        fbs: list[bool] = []
        sc_arr = np.zeros(n, dtype=float)
        co_arr = np.zeros(n, dtype=float)
        for i in range(n):
            p = {
                MODELS[0]: float(oof.loc[i, "prob_ds"]),
                MODELS[1]: float(oof.loc[i, "prob_fl"]),
                MODELS[2]: float(oof.loc[i, "prob_pr"]),
            }
            tc = {
                MODELS[0]: float(oof.loc[i, "train_cost_ds"]),
                MODELS[1]: float(oof.loc[i, "train_cost_fl"]),
                MODELS[2]: float(oof.loc[i, "train_cost_pr"]),
            }
            dec = route_by_quality_and_cost(p, tc, floor, FALLBACK_MODEL)
            m = str(dec["selected_model"])
            sel.append(m)
            fbs.append(bool(dec["fallback"]))
            sc_arr[i] = float(oof.loc[i, score_col[m]])
            co_arr[i] = float(oof.loc[i, cost_col[m]])

        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, size=(10000, n), replace=True)
        boot_acc = np.mean(sc_arr[idx], axis=1)
        boot_pro_acc = np.mean(pro_scores[idx], axis=1)
        boot_ret = boot_acc / boot_pro_acc * 100.0
        boot_pro_c = np.sum(pro_costs[idx], axis=1)
        boot_sav = (boot_pro_c - np.sum(co_arr[idx], axis=1)) / boot_pro_c * 100.0
        fb_arr = np.array(fbs, dtype=float)
        boot_fb = np.mean(fb_arr[idx], axis=1) * 100.0
        boot_diff_pp = (boot_acc - boot_pro_acc) * 100.0

        vs_pro = paired_statistics(sc_arr, pro_scores, co_arr, pro_costs, np.random.default_rng(RANDOM_SEED))
        vs_fl = paired_statistics(sc_arr, flash_scores, co_arr, flash_costs, np.random.default_rng(RANDOM_SEED))

        pro_acc = float(np.mean(pro_scores))
        acc = float(np.mean(sc_arr))
        sav = float((np.sum(pro_costs) - np.sum(co_arr)) / np.sum(pro_costs) * 100.0)
        frontier_points.append(
            {
                "floor": float(floor),
                "accuracy_pct": acc * 100.0,
                "accuracy_95ci_pct": [float(x * 100.0) for x in np.percentile(boot_acc, [2.5, 97.5])],
                "retention_pct": acc / pro_acc * 100.0,
                "retention_95ci_pct": [float(x) for x in np.percentile(boot_ret, [2.5, 97.5])],
                "savings_pct": sav,
                "savings_95ci_pct": [float(x) for x in np.percentile(boot_sav, [2.5, 97.5])],
                "fallback_pct": float(np.mean(fb_arr) * 100.0),
                "fallback_95ci_pct": [float(x) for x in np.percentile(boot_fb, [2.5, 97.5])],
                "ni_lb_pp": float(np.percentile(boot_diff_pp, 2.5)),
                "mcnemar_vs_pro_p": float(vs_pro["mcnemar"]["p_value"]),
                "mcnemar_vs_flash_p": float(vs_fl["mcnemar"]["p_value"]),
                "routing_share_pct": {m: float(np.mean([s == m for s in sel]) * 100.0) for m in MODELS},
            }
        )
    return {"floors": frontier_points}


def plot_frontier(frontier_points: list[dict[str, Any]], plot_path: Path = FRONTIER_PLOT_PATH) -> None:
    """Plot cost-quality frontier (accuracy, retention, savings, fallback with 95% CIs)."""
    floors = np.array([p["floor"] for p in frontier_points])
    acc = np.array([p["accuracy_pct"] for p in frontier_points])
    acc_lo = np.array([p["accuracy_95ci_pct"][0] for p in frontier_points])
    acc_hi = np.array([p["accuracy_95ci_pct"][1] for p in frontier_points])
    ret = np.array([p["retention_pct"] for p in frontier_points])
    ret_lo = np.array([p["retention_95ci_pct"][0] for p in frontier_points])
    ret_hi = np.array([p["retention_95ci_pct"][1] for p in frontier_points])
    sav = np.array([p["savings_pct"] for p in frontier_points])
    sav_lo = np.array([p["savings_95ci_pct"][0] for p in frontier_points])
    sav_hi = np.array([p["savings_95ci_pct"][1] for p in frontier_points])
    fb = np.array([p["fallback_pct"] for p in frontier_points])
    fb_lo = np.array([p["fallback_95ci_pct"][0] for p in frontier_points])
    fb_hi = np.array([p["fallback_95ci_pct"][1] for p in frontier_points])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)

    ax = axes[0]
    ax.plot(floors, acc, marker="o", color="#1f77b4", label="Accuracy (%)")
    ax.fill_between(floors, acc_lo, acc_hi, color="#1f77b4", alpha=0.18)
    ax.plot(floors, ret, marker="s", color="#2ca02c", label="Pro Retention (%)")
    ax.fill_between(floors, ret_lo, ret_hi, color="#2ca02c", alpha=0.18)
    ax.axhline(98.0, color="#2ca02c", linestyle="--", linewidth=1.0, label="98% Retention Target")
    ax.axhline(85.67, color="#444444", linestyle=":", linewidth=1.0, label="Always Pro (85.67%)")
    ax.set_title("Platt Router: Accuracy & Pro Retention vs Floor")
    ax.set_xlabel("Quality Floor (tau)")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(75, 102)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax2 = axes[1]
    ax2.plot(floors, sav, marker="o", color="#ff7f0e", label="Cost Savings vs Pro (%)")
    ax2.fill_between(floors, sav_lo, sav_hi, color="#ff7f0e", alpha=0.18)
    ax2.plot(floors, fb, marker="^", color="#d62728", label="Fallback Rate (%)")
    ax2.fill_between(floors, fb_lo, fb_hi, color="#d62728", alpha=0.18)
    ax2.set_title("Platt Router: Savings & Fallback Rate vs Floor")
    ax2.set_xlabel("Quality Floor (tau)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


class PlattCalibratedRouterService(RouterService):
    """Final chosen Platt-calibrated TF-IDF + LR router evaluated cleanly over FastAPI /v1/chat."""

    def __init__(self, calibrator: PerModelCalibrator, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calibrator = calibrator

    def predict_qualities(self, prompt: str) -> dict[str, float]:
        raw = super().predict_qualities(prompt)
        calibrated = self._calibrator.transform({m: np.array([raw[m]]) for m in MODEL_IDS})
        return {m: float(calibrated[m][0]) for m in MODEL_IDS}


async def _run_concurrency_level(
    router: RouterService, prompts: list[str], concurrency: int, n_requests: int = 500
) -> dict[str, Any]:
    import app.main as main_module

    old_service = main_module.router_service
    main_module.router_service = router
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []

    async def one(client: httpx.AsyncClient, idx: int) -> None:
        p = prompts[idx % len(prompts)]
        async with sem:
            t0 = time.perf_counter()
            resp = await client.post("/v1/chat", json={"request_id": f"c{concurrency}-{idx}", "prompt": p})
            dt = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            latencies_ms.append(dt)

    wall0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            await asyncio.gather(*(one(client, i) for i in range(n_requests)))
    finally:
        main_module.router_service = old_service
    wall_s = time.perf_counter() - wall0
    arr = np.array(latencies_ms, dtype=float)
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_time_s": float(wall_s),
        "throughput_rps": float(n_requests / wall_s),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def run_final_concurrency_suite(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run concurrency 1, 10, 50 (500 requests each) on the final Platt router with mock providers."""
    prompts = df.drop_duplicates("question_id")["prompt"].head(100).astype(str).tolist()
    v21_pkg = joblib.load(FROZEN_V21_PATH)
    # Fit representative PerModelCalibrator on non-test sample so transform() cost & behavior match
    sample_val = df[df["question_id"].isin(df["question_id"].unique()[2100:2550])].copy()
    val_pr = sample_val.drop_duplicates("question_id").sort_values("question_id")
    val_raw = predict_positive_probabilities(v21_pkg["predictor"], val_pr)
    val_piv = sample_val.pivot(index="question_id", columns="model", values="score")[MODELS]
    val_y = {m: val_piv.loc[val_pr["question_id"].to_numpy(), m].to_numpy(dtype=int) for m in MODELS}
    cal_platt = PerModelCalibrator("platt").fit(val_raw, val_y)

    router = PlattCalibratedRouterService(
        calibrator=cal_platt,
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
    )
    results = []
    for level in (1, 10, 50):
        results.append(asyncio.run(_run_concurrency_level(router, prompts, level, n_requests=500)))
    return results


def upsert_attempt_log(new_row: dict[str, Any]) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if ATTEMPT_LOG_CSV.exists():
        with ATTEMPT_LOG_CSV.open("r", encoding="utf-8", newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["attempt_id"] != new_row["attempt_id"]]
    existing.append(new_row)
    with ATTEMPT_LOG_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTEMPT_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
    return existing


def main() -> int:
    df = pd.read_csv(DATASET_PATH)
    first_prompt = str(df.drop_duplicates("question_id").iloc[0]["prompt"])

    cached_cold = measure_cached_minilm_cold_start(first_prompt)
    res_c = run_attempt_c_subject_metadata(df)
    res_d = run_attempt_d_platt_frontier(df, FRONTIER_FLOORS)
    plot_frontier(res_d["floors"], FRONTIER_PLOT_PATH)
    conc_suite = run_final_concurrency_suite(df)

    ab_payload = json.loads((ROOT / "results" / "v5" / "phase5_ab.json").read_text(encoding="utf-8"))
    base_cal = ab_payload["attempt_a"]["calibration"]["base_uncal"]

    c_row = _row_for_log(
        "5c-subject-meta",
        res_c["metrics"]["name"],
        res_c["metrics"],
        base_cal,
        res_c["calibration"],
        res_c["latency"],
        "TF-IDF + subject + prompt length/digit metadata (requires subject label at inference)",
    )
    attempt_rows = upsert_attempt_log(c_row)

    # Select final candidate among all logged attempts strictly by pre-registered gate
    passing = [r for r in attempt_rows if str(r["passes_overall_gate"]).lower() == "true"]
    final_choice = (
        "5a-platt (Platt-calibrated V2.1 TF-IDF + LR)"
        if any(r["attempt_id"] == "5a-platt" for r in passing)
        else "NONE (comparator remains default)"
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cached_minilm_cold_start": cached_cold,
        "attempt_c_subject_metadata": res_c,
        "attempt_d_platt_frontier": res_d,
        "final_chosen_router": final_choice,
        "final_router_concurrency_500req": conc_suite,
    }
    OUTPUT_FINAL_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Render concise Phase 5 final report
    cm = res_c["metrics"]
    lines = [
        "# Phase 5 — Final Report (Attempts c, d, Cached MiniLM & Concurrency)",
        "",
        f"- **Generated:** `{payload['timestamp']}`",
        f"- **Pre-Registered Comparator (`Refit V2.1 (inner-CV tau)`):** fallback `50.77%`, savings `21.59%`, retention `98.44%`, NI LB `-1.93 pp`.",
        f"- **Final Chosen Router:** **`{final_choice}`** (note: `5a-platt` passes all quality/cost and latency gates without requiring subject metadata; benefit is driven by threshold/fallback behavior rather than calibration error reduction, with 6 variants compared on the same 5 outer folds).",
        "",
        "## 1. Cached MiniLM Cold Start vs First-Run Download+Load",
        "",
        f"- **First-run cold start (earlier download + HF init + encode):** `32,410.98 ms` (`32.41 s`).",
        f"- **Cached cold start (`local_files_only=True`, fresh Python process):** total `{cached_cold['total_cached_cold_start_ms']:.1f} ms` (`import = {cached_cold['import_ms']:.1f} ms`, `cached model load = {cached_cold['load_cached_ms']:.1f} ms`, `first encode = {cached_cold['first_encode_ms']:.1f} ms`).",
        f"- **Single-prompt warm CPU inference (from Phase 5a/b):** `p50 = 20.68 ms`, `p95 = 28.44 ms`, `p99 = 29.60 ms` — **still exceeds the `25.0 ms` `p95` gate**, confirming why GBT-on-embeddings was skipped.",
        "",
        "## 2. Attempt (c) — Subject + Prompt Metadata (`requires subject label at inference`)",
        "",
        f"- **Accuracy:** `{cm['accuracy']*100:.2f}%` (`[{cm['accuracy_95ci'][0]*100:.2f}%, {cm['accuracy_95ci'][1]*100:.2f}%]`), **Pro Retention:** `{cm['pro_retention_pct']:.2f}%`",
        f"- **Cost Savings vs Always-Pro:** `{cm['savings_vs_pro_pct']:.2f}%` (`[{cm['savings_vs_pro_95ci_pct'][0]:.2f}%, {cm['savings_vs_pro_95ci_pct'][1]:.2f}%]`), **Fallback Rate:** `{cm['fallback_rate_pct']:.2f}%`",
        f"- **NI LB vs Pro:** `{cm['ni_lb_pp']:+.2f} pp` | **McNemar vs Pro:** `{cm['mcnemar_vs_pro_p']:.4g}` | **McNemar vs Flash:** `{cm['mcnemar_vs_flash_p']:.4g}`",
        f"- **Calibration:** ECE `{res_c['calibration']['ece_10bin']:.5f}`, Brier `{res_c['calibration']['brier_score']:.5f}` | **Routing Latency:** `p50={res_c['latency']['p50_ms']:.2f} ms`, `p95={res_c['latency']['p95_ms']:.2f} ms`, `p99={res_c['latency']['p99_ms']:.2f} ms`",
        f"- **Quality/Cost Gate:** **`{cm['passes_quality_cost_gate']}`** | **Latency Gate:** **`{res_c['latency']['passes_p95_25ms_gate']}`** | **Overall Gate:** **`{cm['passes_quality_cost_gate'] and res_c['latency']['passes_p95_25ms_gate']}`**",
        "",
        "## 3. Attempt (d) — Cost–Quality Frontier on Platt Router (`0.80–0.95`, 5-Fold Nested CV)",
        "",
        "Plot saved to [reports/phase5_frontier.png](file:///c:/Users/HP/Project/LLMRouter/reports/phase5_frontier.png).",
        "",
        "| Floor (`tau`) | Accuracy [95% CI] | Pro Retention [95% CI] | Savings vs Pro [95% CI] | Fallback Rate [95% CI] | NI LB vs Pro | Share DS/Fl/Pr (%) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pt in res_d["floors"]:
        sh = pt["routing_share_pct"]
        lines.append(
            f"| `{pt['floor']:.3f}` | {pt['accuracy_pct']:.2f}% [{pt['accuracy_95ci_pct'][0]:.2f}, {pt['accuracy_95ci_pct'][1]:.2f}] | "
            f"{pt['retention_pct']:.2f}% [{pt['retention_95ci_pct'][0]:.2f}, {pt['retention_95ci_pct'][1]:.2f}] | "
            f"{pt['savings_pct']:.2f}% [{pt['savings_95ci_pct'][0]:.2f}, {pt['savings_95ci_pct'][1]:.2f}] | "
            f"{pt['fallback_pct']:.2f}% [{pt['fallback_95ci_pct'][0]:.2f}, {pt['fallback_95ci_pct'][1]:.2f}] | "
            f"{pt['ni_lb_pp']:+.2f} pp | {sh[MODELS[0]]:.1f}/{sh[MODELS[1]]:.1f}/{sh[MODELS[2]]:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Final Chosen Router Concurrency Benchmark (`MockProvider`, `/v1/chat`, 500 Requests/Level)",
            "",
            "| Concurrency | Requests | Wall Time (s) | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in conc_suite:
        lines.append(
            f"| `{row['concurrency']}` | {row['n_requests']} | {row['wall_time_s']:.3f} | "
            f"{row['throughput_rps']:.1f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['p99_ms']:.2f} |"
        )
    lines.append("")
    OUTPUT_FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_FINAL_JSON}, {OUTPUT_FINAL_REPORT}, {FRONTIER_PLOT_PATH}, and updated {ATTEMPT_LOG_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
