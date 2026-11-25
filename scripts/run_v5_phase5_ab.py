"""LLMRouter V5 Phase 5 — Attempts (a) & (b) + Routing-Latency Benchmark.

Evaluates strictly on the same 5-fold prompt-grouped nested CV (N=3,000 MMLU-Pro
test_3000, GroupKFold by question_id) used in Phase 1, comparing against the
pre-registered comparator `Refit V2.1 (inner-CV tau)`:
  - fallback = 50.77%, savings = 21.59%, retention = 98.44%, NI LB = -1.93 pp
  - Gate: savings > 21.59% AND fallback < 50.77% AND retention >= 98.0%
          AND NI LB > -2.00 pp AND routing-decision p95 <= 25.0 ms

Attempts executed here:
  a) Probability calibration of the existing LR (Platt & Isotonic, fit on inner
     validation only) + ECE/Brier before & after.
  b) Sentence-transformer embeddings (`all-MiniLM-L6-v2`) alone and concatenated
     with TF-IDF.

Logs all attempts to `results/v5/attempt_log.csv` and writes
`results/v5/phase5_ab.json` + `reports/phase5_ab.md`.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.main import app  # noqa: E402
from app.providers.factory import ProviderFactory  # noqa: E402
from app.services.inference_service import InferenceService  # noqa: E402
from app.services.router_service import RouterService  # noqa: E402
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
    THRESHOLDS,
    make_v21_predictor,
    paired_statistics,
    predict_positive_probabilities,
    route_with_policy,
)

ATTEMPT_LOG_CSV = ROOT / "results" / "v5" / "attempt_log.csv"
OUTPUT_AB_JSON = ROOT / "results" / "v5" / "phase5_ab.json"
OUTPUT_AB_REPORT = ROOT / "reports" / "phase5_ab.md"

COMPARATOR = {
    "name": "Refit V2.1 (inner-CV tau)",
    "fallback_pct": 50.77,
    "savings_pct": 21.59,
    "retention_pct": 98.44,
    "ni_lb_pp": -1.93,
}
LATENCY_P95_LIMIT_MS = 25.0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ATTEMPT_LOG_FIELDS = [
    "attempt_id",
    "method",
    "accuracy_pct",
    "retention_pct",
    "savings_pct",
    "fallback_pct",
    "ni_lb_pp",
    "mcnemar_vs_pro_p",
    "mcnemar_vs_flash_p",
    "ece_before",
    "ece_after",
    "brier_before",
    "brier_after",
    "routing_p50_ms",
    "routing_p95_ms",
    "routing_p99_ms",
    "cold_start_ms",
    "passes_quality_cost_gate",
    "passes_latency_gate",
    "passes_overall_gate",
    "notes",
]


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute 10-bin Expected Calibration Error over [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        acc_bin = float(np.mean(y_true[mask]))
        conf_bin = float(np.mean(y_prob[mask]))
        ece += (count / n) * abs(acc_bin - conf_bin)
    return float(ece)


def _prob_to_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


class PerModelCalibrator:
    """Fit per-model Platt (sigmoid) or Isotonic calibration strictly on inner validation."""

    def __init__(self, method: str) -> None:
        if method not in {"platt", "isotonic"}:
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self._calibrators: dict[str, Any] = {}

    def fit(
        self, probs_by_model: dict[str, np.ndarray], y_by_model: dict[str, np.ndarray]
    ) -> "PerModelCalibrator":
        for model in MODELS:
            p = np.asarray(probs_by_model[model], dtype=float)
            y = np.asarray(y_by_model[model], dtype=int)
            if self.method == "platt":
                lr = LogisticRegression(C=1e6, max_iter=1000, random_state=RANDOM_SEED)
                lr.fit(_prob_to_logit(p).reshape(-1, 1), y)
                self._calibrators[model] = lr
            else:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(p, y)
                self._calibrators[model] = iso
        return self

    def transform(self, probs_by_model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for model in MODELS:
            p = np.asarray(probs_by_model[model], dtype=float)
            cal = self._calibrators[model]
            if self.method == "platt":
                out[model] = cal.predict_proba(_prob_to_logit(p).reshape(-1, 1))[:, 1]
            else:
                out[model] = np.clip(cal.predict(p), 0.0, 1.0)
        return out


def _cross_fit_calibrate_inner_val(
    inner_val_probs: dict[str, np.ndarray],
    inner_val_y: dict[str, np.ndarray],
    qids: np.ndarray,
    method: str,
) -> dict[str, np.ndarray]:
    """Produce leakage-free OOF calibrated probabilities inside inner_val for tau selection."""
    n = len(qids)
    oof_cal = {m: np.zeros(n, dtype=float) for m in MODELS}
    gkf = GroupKFold(n_splits=5)
    for tr_idx, te_idx in gkf.split(np.zeros(n), groups=qids):
        cal = PerModelCalibrator(method).fit(
            {m: inner_val_probs[m][tr_idx] for m in MODELS},
            {m: inner_val_y[m][tr_idx] for m in MODELS},
        )
        pred = cal.transform({m: inner_val_probs[m][te_idx] for m in MODELS})
        for m in MODELS:
            oof_cal[m][te_idx] = pred[m]
    return oof_cal


def _select_tau_from_probs(
    probs_by_model: dict[str, np.ndarray],
    train_costs: dict[str, float],
    score_pivot: pd.DataFrame,
    cost_pivot: pd.DataFrame,
    qids: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    pro_acc = float(score_pivot.loc[qids, FALLBACK_MODEL].mean())
    pro_cost = float(cost_pivot.loc[qids, FALLBACK_MODEL].sum())
    target_acc = 0.98 * pro_acc
    sweep: list[dict[str, Any]] = []
    for tau in THRESHOLDS:
        chosen, fbs = route_with_policy(probs_by_model, train_costs, tau, FALLBACK_MODEL)
        scores = np.array([float(score_pivot.loc[q, m]) for q, m in zip(qids, chosen)])
        costs = np.array([float(cost_pivot.loc[q, m]) for q, m in zip(qids, chosen)])
        acc = float(scores.mean())
        tot_cost = float(costs.sum())
        savings_pct = float((pro_cost - tot_cost) / pro_cost * 100.0) if pro_cost > 0 else 0.0
        sweep.append(
            {
                "threshold": float(tau),
                "accuracy": acc,
                "retention_pct": float(acc / pro_acc * 100.0) if pro_acc > 0 else 0.0,
                "avg_cost": float(costs.mean()),
                "savings_vs_pro_pct": savings_pct,
                "fallback_rate_pct": float(np.mean(fbs) * 100.0),
                "meets_98pct_target": acc >= target_acc,
            }
        )
    eligible = [row for row in sweep if row["meets_98pct_target"]]
    if eligible:
        max_saving = max(row["savings_vs_pro_pct"] for row in eligible)
        tied = [row for row in eligible if np.isclose(row["savings_vs_pro_pct"], max_saving)]
        best = min(tied, key=lambda row: (row["avg_cost"], -row["threshold"]))
    else:
        best = max(sweep, key=lambda row: (row["accuracy"], row["threshold"]))
    return float(best["threshold"]), {"best": best}


def _summarize_candidate(
    name: str,
    oof_scores: np.ndarray,
    oof_costs: np.ndarray,
    oof_selected: list[str],
    oof_fallbacks: list[bool],
    pro_scores: np.ndarray,
    pro_costs: np.ndarray,
    flash_scores: np.ndarray,
    flash_costs: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    acc = float(np.mean(oof_scores))
    pro_acc = float(np.mean(pro_scores))
    pro_total = float(np.sum(pro_costs))
    tot_cost = float(np.sum(oof_costs))
    retention_pct = float(acc / pro_acc * 100.0) if pro_acc > 0 else 0.0
    savings_pct = float((pro_total - tot_cost) / pro_total * 100.0) if pro_total > 0 else 0.0
    fallback_pct = float(np.mean(oof_fallbacks) * 100.0)

    vs_pro = paired_statistics(oof_scores, pro_scores, oof_costs, pro_costs, rng)
    vs_flash = paired_statistics(oof_scores, flash_scores, oof_costs, flash_costs, rng)
    ni_lb_pp = float(vs_pro["bootstrap_95ci"]["accuracy_diff_pp_ci"][0])

    shares = {m: float(np.mean([s == m for s in oof_selected]) * 100.0) for m in MODELS}
    passes_qc = (
        savings_pct > COMPARATOR["savings_pct"]
        and fallback_pct < COMPARATOR["fallback_pct"]
        and retention_pct >= 98.0
        and ni_lb_pp > NON_INFERIORITY_MARGIN_PP
    )
    return {
        "name": name,
        "accuracy": acc,
        "accuracy_95ci": vs_pro["bootstrap_95ci"]["accuracy_a_ci"],
        "pro_retention_pct": retention_pct,
        "savings_vs_pro_pct": savings_pct,
        "savings_vs_pro_95ci_pct": vs_pro["bootstrap_95ci"]["cost_savings_pct_ci"],
        "fallback_rate_pct": fallback_pct,
        "ni_lb_pp": ni_lb_pp,
        "mcnemar_vs_pro_p": float(vs_pro["mcnemar"]["p_value"]),
        "mcnemar_vs_flash_p": float(vs_flash["mcnemar"]["p_value"]),
        "routing_share_pct": shares,
        "passes_quality_cost_gate": bool(passes_qc),
    }


def run_attempt_a_calibration(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate Platt and Isotonic calibration (fit strictly on inner_val) across 5 outer folds."""
    df_sorted = df.sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = df_sorted.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)

    buckets: dict[str, dict[str, list[Any]]] = {
        key: {"scores": [], "costs": [], "selected": [], "fallback": [], "y_true": [], "y_prob": []}
        for key in ("base_uncal", "platt", "isotonic")
    }
    pro_scores_all: list[float] = []
    pro_costs_all: list[float] = []
    flash_scores_all: list[float] = []
    flash_costs_all: list[float] = []
    fold_details: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(q_df, groups=q_df["question_id"]), start=1):
        train_qids = set(map(int, q_df.iloc[tr_idx]["question_id"]))
        test_qids = set(map(int, q_df.iloc[te_idx]["question_id"]))
        assert train_qids.isdisjoint(test_qids)

        outer_train = df_sorted[df_sorted["question_id"].isin(train_qids)].copy()
        outer_test = df_sorted[df_sorted["question_id"].isin(test_qids)].copy()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED + fold_idx)
        inner_q = q_df.iloc[tr_idx].reset_index(drop=True)
        in_tr_i, in_val_i = next(gss.split(inner_q, groups=inner_q["question_id"]))
        in_tr_qids = set(map(int, inner_q.iloc[in_tr_i]["question_id"]))
        in_val_qids = set(map(int, inner_q.iloc[in_val_i]["question_id"]))

        inner_train = outer_train[outer_train["question_id"].isin(in_tr_qids)].copy()
        inner_val = outer_train[outer_train["question_id"].isin(in_val_qids)].copy()

        base_pipe = make_v21_predictor()
        base_pipe.fit(inner_train[["prompt", "model"]], inner_train["score"].astype(int))
        train_costs = inner_train.groupby("model")["cost"].mean().to_dict()

        val_prompts = inner_val.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        val_qids_arr = val_prompts["question_id"].to_numpy()
        val_scores_piv = inner_val.pivot(index="question_id", columns="model", values="score")[MODELS]
        val_costs_piv = inner_val.pivot(index="question_id", columns="model", values="cost")[MODELS]

        val_raw_probs = predict_positive_probabilities(base_pipe, val_prompts)
        val_y = {m: val_scores_piv.loc[val_qids_arr, m].to_numpy(dtype=int) for m in MODELS}

        # Select tau on uncalibrated inner_val, and on OOF cross-fitted inner_val calibrated probs
        tau_uncal, _ = _select_tau_from_probs(val_raw_probs, train_costs, val_scores_piv, val_costs_piv, val_qids_arr)
        val_platt_oof = _cross_fit_calibrate_inner_val(val_raw_probs, val_y, val_qids_arr, "platt")
        tau_platt, _ = _select_tau_from_probs(val_platt_oof, train_costs, val_scores_piv, val_costs_piv, val_qids_arr)
        val_iso_oof = _cross_fit_calibrate_inner_val(val_raw_probs, val_y, val_qids_arr, "isotonic")
        tau_iso, _ = _select_tau_from_probs(val_iso_oof, train_costs, val_scores_piv, val_costs_piv, val_qids_arr)

        # Fit final calibrators on ALL inner_val (480 prompts)
        cal_platt = PerModelCalibrator("platt").fit(val_raw_probs, val_y)
        cal_iso = PerModelCalibrator("isotonic").fit(val_raw_probs, val_y)

        # Evaluate on held-out outer_test
        test_prompts = outer_test.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        test_qids_arr = test_prompts["question_id"].to_numpy()
        test_scores_piv = outer_test.pivot(index="question_id", columns="model", values="score")[MODELS]
        test_costs_piv = outer_test.pivot(index="question_id", columns="model", values="cost")[MODELS]

        raw_test_probs = predict_positive_probabilities(base_pipe, test_prompts)
        platt_test_probs = cal_platt.transform(raw_test_probs)
        iso_test_probs = cal_iso.transform(raw_test_probs)

        fold_details.append({"fold": fold_idx, "tau_uncal": tau_uncal, "tau_platt": tau_platt, "tau_isotonic": tau_iso})

        for qid in test_qids_arr:
            pro_scores_all.append(float(test_scores_piv.loc[int(qid), FALLBACK_MODEL]))
            pro_costs_all.append(float(test_costs_piv.loc[int(qid), FALLBACK_MODEL]))
            flash_scores_all.append(float(test_scores_piv.loc[int(qid), MODELS[1]]))
            flash_costs_all.append(float(test_costs_piv.loc[int(qid), MODELS[1]]))

        for key, probs_dict, tau in [
            ("base_uncal", raw_test_probs, tau_uncal),
            ("platt", platt_test_probs, tau_platt),
            ("isotonic", iso_test_probs, tau_iso),
        ]:
            sel, fb = route_with_policy(probs_dict, train_costs, tau, FALLBACK_MODEL)
            for i, qid in enumerate(test_qids_arr):
                buckets[key]["scores"].append(float(test_scores_piv.loc[int(qid), sel[i]]))
                buckets[key]["costs"].append(float(test_costs_piv.loc[int(qid), sel[i]]))
                buckets[key]["selected"].append(sel[i])
                buckets[key]["fallback"].append(bool(fb[i]))
            for m in MODELS:
                buckets[key]["y_true"].extend(test_scores_piv.loc[test_qids_arr, m].to_numpy(dtype=int).tolist())
                buckets[key]["y_prob"].extend(probs_dict[m].tolist())

    pro_s = np.array(pro_scores_all)
    pro_c = np.array(pro_costs_all)
    fl_s = np.array(flash_scores_all)
    fl_c = np.array(flash_costs_all)

    calibration_stats = {}
    metrics = {}
    for key, label in [
        ("base_uncal", "Base LR (inner-train only, uncalibrated)"),
        ("platt", "Attempt 5a-1: Platt Calibration (inner-val fit)"),
        ("isotonic", "Attempt 5a-2: Isotonic Calibration (inner-val fit)"),
    ]:
        yt = np.array(buckets[key]["y_true"], dtype=int)
        yp = np.array(buckets[key]["y_prob"], dtype=float)
        calibration_stats[key] = {
            "ece_10bin": compute_ece(yt, yp, n_bins=10),
            "brier_score": float(brier_score_loss(yt, yp)),
        }
        metrics[key] = _summarize_candidate(
            label,
            np.array(buckets[key]["scores"]),
            np.array(buckets[key]["costs"]),
            buckets[key]["selected"],
            buckets[key]["fallback"],
            pro_s,
            pro_c,
            fl_s,
            fl_c,
        )

    return {"folds": fold_details, "calibration": calibration_stats, "metrics": metrics}


def _fit_predict_embedding_model(
    train_df: pd.DataFrame,
    test_prompts_df: pd.DataFrame,
    emb_by_qid: dict[int, np.ndarray],
    *,
    include_tfidf: bool,
) -> dict[str, np.ndarray]:
    """Fit LogisticRegression on MiniLM embeddings (+ optional TF-IDF) + model OHE."""
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_model_tr = ohe.fit_transform(train_df[["model"]])
    X_emb_tr = csr_matrix(np.vstack([emb_by_qid[int(q)] for q in train_df["question_id"]]))
    if include_tfidf:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
        X_txt_tr = tfidf.fit_transform(train_df["prompt"])
        X_tr = hstack([X_txt_tr, X_emb_tr, X_model_tr]).tocsr()
    else:
        tfidf = None
        X_tr = hstack([X_emb_tr, X_model_tr]).tocsr()

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_tr, train_df["score"].astype(int))
    pos_idx = list(clf.classes_).index(1)

    X_emb_te = csr_matrix(np.vstack([emb_by_qid[int(q)] for q in test_prompts_df["question_id"]]))
    X_txt_te = tfidf.transform(test_prompts_df["prompt"]) if tfidf is not None else None
    probs: dict[str, np.ndarray] = {}
    for m in MODELS:
        X_m_te = ohe.transform(pd.DataFrame({"model": [m] * len(test_prompts_df)}))
        X_te = hstack([X_txt_te, X_emb_te, X_m_te]).tocsr() if X_txt_te is not None else hstack([X_emb_te, X_m_te]).tocsr()
        probs[m] = clf.predict_proba(X_te)[:, pos_idx]
    return probs


def run_attempt_b_embeddings(
    df: pd.DataFrame, emb_by_qid: dict[int, np.ndarray]
) -> dict[str, Any]:
    """Evaluate MiniLM-only and MiniLM+TF-IDF across the same 5-fold nested CV."""
    df_sorted = df.sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = df_sorted.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)

    buckets: dict[str, dict[str, list[Any]]] = {
        key: {"scores": [], "costs": [], "selected": [], "fallback": [], "y_true": [], "y_prob": []}
        for key in ("minilm_only", "minilm_plus_tfidf")
    }
    pro_scores_all: list[float] = []
    pro_costs_all: list[float] = []
    flash_scores_all: list[float] = []
    flash_costs_all: list[float] = []
    fold_details: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(q_df, groups=q_df["question_id"]), start=1):
        train_qids = set(map(int, q_df.iloc[tr_idx]["question_id"]))
        test_qids = set(map(int, q_df.iloc[te_idx]["question_id"]))
        assert train_qids.isdisjoint(test_qids)

        outer_train = df_sorted[df_sorted["question_id"].isin(train_qids)].copy()
        outer_test = df_sorted[df_sorted["question_id"].isin(test_qids)].copy()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED + fold_idx)
        inner_q = q_df.iloc[tr_idx].reset_index(drop=True)
        in_tr_i, in_val_i = next(gss.split(inner_q, groups=inner_q["question_id"]))
        in_tr_qids = set(map(int, inner_q.iloc[in_tr_i]["question_id"]))
        in_val_qids = set(map(int, inner_q.iloc[in_val_i]["question_id"]))

        inner_train = outer_train[outer_train["question_id"].isin(in_tr_qids)].copy()
        inner_val = outer_train[outer_train["question_id"].isin(in_val_qids)].copy()

        val_prompts = inner_val.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        val_qids_arr = val_prompts["question_id"].to_numpy()
        val_scores_piv = inner_val.pivot(index="question_id", columns="model", values="score")[MODELS]
        val_costs_piv = inner_val.pivot(index="question_id", columns="model", values="cost")[MODELS]

        test_prompts = outer_test.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        test_qids_arr = test_prompts["question_id"].to_numpy()
        test_scores_piv = outer_test.pivot(index="question_id", columns="model", values="score")[MODELS]
        test_costs_piv = outer_test.pivot(index="question_id", columns="model", values="cost")[MODELS]

        for qid in test_qids_arr:
            pro_scores_all.append(float(test_scores_piv.loc[int(qid), FALLBACK_MODEL]))
            pro_costs_all.append(float(test_costs_piv.loc[int(qid), FALLBACK_MODEL]))
            flash_scores_all.append(float(test_scores_piv.loc[int(qid), MODELS[1]]))
            flash_costs_all.append(float(test_costs_piv.loc[int(qid), MODELS[1]]))

        in_costs = inner_train.groupby("model")["cost"].mean().to_dict()
        out_costs = outer_train.groupby("model")["cost"].mean().to_dict()

        taus_fold: dict[str, float] = {}
        for key, inc_tfidf in [("minilm_only", False), ("minilm_plus_tfidf", True)]:
            val_probs = _fit_predict_embedding_model(inner_train, val_prompts, emb_by_qid, include_tfidf=inc_tfidf)
            tau_star, _ = _select_tau_from_probs(val_probs, in_costs, val_scores_piv, val_costs_piv, val_qids_arr)
            taus_fold[key] = tau_star

            out_probs = _fit_predict_embedding_model(outer_train, test_prompts, emb_by_qid, include_tfidf=inc_tfidf)
            sel, fb = route_with_policy(out_probs, out_costs, tau_star, FALLBACK_MODEL)
            for i, qid in enumerate(test_qids_arr):
                buckets[key]["scores"].append(float(test_scores_piv.loc[int(qid), sel[i]]))
                buckets[key]["costs"].append(float(test_costs_piv.loc[int(qid), sel[i]]))
                buckets[key]["selected"].append(sel[i])
                buckets[key]["fallback"].append(bool(fb[i]))
            for m in MODELS:
                buckets[key]["y_true"].extend(test_scores_piv.loc[test_qids_arr, m].to_numpy(dtype=int).tolist())
                buckets[key]["y_prob"].extend(out_probs[m].tolist())
        fold_details.append({"fold": fold_idx, **taus_fold})

    pro_s = np.array(pro_scores_all)
    pro_c = np.array(pro_costs_all)
    fl_s = np.array(flash_scores_all)
    fl_c = np.array(flash_costs_all)

    cal = {}
    metrics = {}
    for key, label in [
        ("minilm_only", "Attempt 5b-1: all-MiniLM-L6-v2 + LR"),
        ("minilm_plus_tfidf", "Attempt 5b-2: all-MiniLM-L6-v2 + TF-IDF + LR"),
    ]:
        yt = np.array(buckets[key]["y_true"], dtype=int)
        yp = np.array(buckets[key]["y_prob"], dtype=float)
        cal[key] = {
            "ece_10bin": compute_ece(yt, yp, n_bins=10),
            "brier_score": float(brier_score_loss(yt, yp)),
        }
        metrics[key] = _summarize_candidate(
            label,
            np.array(buckets[key]["scores"]),
            np.array(buckets[key]["costs"]),
            buckets[key]["selected"],
            buckets[key]["fallback"],
            pro_s,
            pro_c,
            fl_s,
            fl_c,
        )
    return {"folds": fold_details, "calibration": cal, "metrics": metrics}


async def _concurrency_benchmark(prompts: list[str], router: RouterService, concurrency: int = 10) -> dict[str, float]:
    import app.main as main_module

    old_service = main_module.router_service
    main_module.router_service = router
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []

    async def one(client: httpx.AsyncClient, idx: int, prompt: str) -> None:
        async with sem:
            t0 = time.perf_counter()
            resp = await client.post("/v1/chat", json={"request_id": f"bench-{idx}", "prompt": prompt})
            dt = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            latencies_ms.append(dt)

    wall0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            await asyncio.gather(*(one(client, i, p) for i, p in enumerate(prompts)))
    finally:
        main_module.router_service = old_service
    wall_s = time.perf_counter() - wall0
    arr = np.array(latencies_ms, dtype=float)
    return {
        "n_requests": len(prompts),
        "concurrency": concurrency,
        "wall_time_s": float(wall_s),
        "throughput_rps": float(len(prompts) / wall_s),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def measure_routing_latencies(df: pd.DataFrame) -> dict[str, Any]:
    """Measure cold start, single-prompt routing p50/p95/p99, and mock concurrency."""
    prompts = df.drop_duplicates("question_id")["prompt"].head(100).astype(str).tolist()
    sample_train = df[df["question_id"].isin(df["question_id"].unique()[:500])].copy()
    sample_val = df[df["question_id"].isin(df["question_id"].unique()[500:650])].copy()

    # 1. Baseline Frozen V2.1 (TF-IDF + LR) cold start + per-request routing latency
    t0 = time.perf_counter()
    router_v21 = RouterService(
        inference_service=InferenceService(ProviderFactory(mode="mock")),
        telemetry_service=TelemetryService(NoopTelemetryRepository()),
    )
    _ = router_v21.predict_qualities(prompts[0])
    v21_cold_ms = (time.perf_counter() - t0) * 1000.0

    v21_lats: list[float] = []
    for p in prompts:
        s0 = time.perf_counter()
        probs = router_v21.predict_qualities(p)
        _ = route_by_quality_and_cost(probs, router_v21.training_average_costs, router_v21.threshold, FALLBACK_MODEL)
        v21_lats.append((time.perf_counter() - s0) * 1000.0)
    conc_v21 = asyncio.run(_concurrency_benchmark(prompts[:50], router_v21, concurrency=10))

    # 2. Attempt (a) TF-IDF + LR + Platt/Isotonic overhead
    v21_pkg = joblib.load(FROZEN_V21_PATH)
    val_pr = sample_val.drop_duplicates("question_id").sort_values("question_id")
    val_raw = predict_positive_probabilities(v21_pkg["predictor"], val_pr)
    val_piv = sample_val.pivot(index="question_id", columns="model", values="score")[MODELS]
    val_y = {m: val_piv.loc[val_pr["question_id"].to_numpy(), m].to_numpy(dtype=int) for m in MODELS}
    cal_platt = PerModelCalibrator("platt").fit(val_raw, val_y)
    cal_iso = PerModelCalibrator("isotonic").fit(val_raw, val_y)

    platt_lats: list[float] = []
    iso_lats: list[float] = []
    for p in prompts:
        s0 = time.perf_counter()
        pr = router_v21.predict_qualities(p)
        cp = cal_platt.transform({m: np.array([v]) for m, v in pr.items()})
        _ = route_by_quality_and_cost({m: float(cp[m][0]) for m in MODELS}, router_v21.training_average_costs, router_v21.threshold, FALLBACK_MODEL)
        platt_lats.append((time.perf_counter() - s0) * 1000.0)

        s1 = time.perf_counter()
        pr2 = router_v21.predict_qualities(p)
        ci = cal_iso.transform({m: np.array([v]) for m, v in pr2.items()})
        _ = route_by_quality_and_cost({m: float(ci[m][0]) for m in MODELS}, router_v21.training_average_costs, router_v21.threshold, FALLBACK_MODEL)
        iso_lats.append((time.perf_counter() - s1) * 1000.0)

    # 3. Attempt (b) MiniLM cold start + single-prompt encoding + LR latency
    from sentence_transformers import SentenceTransformer

    st_t0 = time.perf_counter()
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _ = st_model.encode([prompts[0]], normalize_embeddings=True, show_progress_bar=False)
    st_cold_ms = (time.perf_counter() - st_t0) * 1000.0

    # Fit tiny representative LR for realistic inference latency timing
    q_unique = sample_train.drop_duplicates("question_id")
    embs_sample = st_model.encode(q_unique["prompt"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    emb_map = {int(q): embs_sample[i] for i, q in enumerate(q_unique["question_id"])}
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_m = ohe.fit_transform(sample_train[["model"]])
    X_e = csr_matrix(np.vstack([emb_map[int(q)] for q in sample_train["question_id"]]))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    X_t = tfidf.fit_transform(sample_train["prompt"])
    clf_emb = LogisticRegression(C=1.0, max_iter=500).fit(hstack([X_e, X_m]).tocsr(), sample_train["score"])
    clf_both = LogisticRegression(C=1.0, max_iter=500).fit(hstack([X_t, X_e, X_m]).tocsr(), sample_train["score"])
    m_ohe_all = ohe.transform(pd.DataFrame({"model": MODELS}))

    minilm_lats: list[float] = []
    minilm_tfidf_lats: list[float] = []
    for p in prompts:
        s0 = time.perf_counter()
        vec = st_model.encode([p], normalize_embeddings=True, show_progress_bar=False)
        X_in = hstack([csr_matrix(np.repeat(vec, 3, axis=0)), m_ohe_all]).tocsr()
        pr_arr = clf_emb.predict_proba(X_in)[:, 1]
        _ = route_by_quality_and_cost({m: float(pr_arr[i]) for i, m in enumerate(MODELS)}, router_v21.training_average_costs, 0.90, FALLBACK_MODEL)
        minilm_lats.append((time.perf_counter() - s0) * 1000.0)

        s1 = time.perf_counter()
        vec2 = st_model.encode([p], normalize_embeddings=True, show_progress_bar=False)
        xt = tfidf.transform([p])
        X_in2 = hstack([vstack([xt, xt, xt]), csr_matrix(np.repeat(vec2, 3, axis=0)), m_ohe_all]).tocsr()
        pr_arr2 = clf_both.predict_proba(X_in2)[:, 1]
        _ = route_by_quality_and_cost({m: float(pr_arr2[i]) for i, m in enumerate(MODELS)}, router_v21.training_average_costs, 0.90, FALLBACK_MODEL)
        minilm_tfidf_lats.append((time.perf_counter() - s1) * 1000.0)

    def _stats(lats: list[float], cold_ms: float) -> dict[str, Any]:
        arr = np.array(lats, dtype=float)
        p95 = float(np.percentile(arr, 95))
        return {
            "cold_start_ms": float(cold_ms),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": p95,
            "p99_ms": float(np.percentile(arr, 99)),
            "passes_p95_25ms_gate": p95 <= LATENCY_P95_LIMIT_MS,
        }

    return {
        "n_timed_prompts": len(prompts),
        "v21_tfidf_lr": _stats(v21_lats, v21_cold_ms),
        "v21_mock_concurrency": conc_v21,
        "a_platt": _stats(platt_lats, v21_cold_ms),
        "a_isotonic": _stats(iso_lats, v21_cold_ms),
        "b_minilm_only": _stats(minilm_lats, st_cold_ms),
        "b_minilm_plus_tfidf": _stats(minilm_tfidf_lats, st_cold_ms),
        "st_model": st_model,
    }


def _row_for_log(
    attempt_id: str,
    method: str,
    m: dict[str, Any],
    cal_before: dict[str, float] | None,
    cal_after: dict[str, float] | None,
    lat: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    passes_lat = bool(lat["passes_p95_25ms_gate"])
    passes_qc = bool(m["passes_quality_cost_gate"])
    return {
        "attempt_id": attempt_id,
        "method": method,
        "accuracy_pct": round(m["accuracy"] * 100.0, 2),
        "retention_pct": round(m["pro_retention_pct"], 2),
        "savings_pct": round(m["savings_vs_pro_pct"], 2),
        "fallback_pct": round(m["fallback_rate_pct"], 2),
        "ni_lb_pp": round(m["ni_lb_pp"], 2),
        "mcnemar_vs_pro_p": f"{m['mcnemar_vs_pro_p']:.4g}",
        "mcnemar_vs_flash_p": f"{m['mcnemar_vs_flash_p']:.4g}",
        "ece_before": round(cal_before["ece_10bin"], 5) if cal_before else "",
        "ece_after": round(cal_after["ece_10bin"], 5) if cal_after else "",
        "brier_before": round(cal_before["brier_score"], 5) if cal_before else "",
        "brier_after": round(cal_after["brier_score"], 5) if cal_after else "",
        "routing_p50_ms": round(lat["p50_ms"], 2),
        "routing_p95_ms": round(lat["p95_ms"], 2),
        "routing_p99_ms": round(lat["p99_ms"], 2),
        "cold_start_ms": round(lat["cold_start_ms"], 2),
        "passes_quality_cost_gate": passes_qc,
        "passes_latency_gate": passes_lat,
        "passes_overall_gate": passes_qc and passes_lat,
        "notes": notes,
    }


def compute_paired_bootstrap_diff(
    a_scores: np.ndarray,
    a_costs: np.ndarray,
    a_fallbacks: np.ndarray,
    b_scores: np.ndarray,
    b_costs: np.ndarray,
    b_fallbacks: np.ndarray,
    pro_costs: np.ndarray,
    *,
    rounds: int = 10000,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Paired bootstrap (A minus B) for accuracy (pp), savings vs Pro (pp), and fallback rate (pp)."""
    n = len(a_scores)
    assert len(b_scores) == n and len(pro_costs) == n
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=(rounds, n), replace=True)

    acc_diff = (np.mean(a_scores[idx], axis=1) - np.mean(b_scores[idx], axis=1)) * 100.0
    pro_sum = np.sum(pro_costs[idx], axis=1)
    sav_a = np.where(pro_sum > 0, (pro_sum - np.sum(a_costs[idx], axis=1)) / pro_sum * 100.0, 0.0)
    sav_b = np.where(pro_sum > 0, (pro_sum - np.sum(b_costs[idx], axis=1)) / pro_sum * 100.0, 0.0)
    sav_diff = sav_a - sav_b
    fb_diff = (
        np.mean(a_fallbacks.astype(float)[idx], axis=1)
        - np.mean(b_fallbacks.astype(float)[idx], axis=1)
    ) * 100.0

    def _entry(pt: float, boot: np.ndarray) -> dict[str, Any]:
        ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        return {
            "point_diff_pp": float(pt),
            "ci_95_pp": ci,
            "excludes_zero": bool(ci[0] > 0.0 or ci[1] < 0.0),
        }

    pt_acc = float((np.mean(a_scores) - np.mean(b_scores)) * 100.0)
    tot_pro = float(np.sum(pro_costs))
    pt_sav = (
        ((tot_pro - float(np.sum(a_costs))) - (tot_pro - float(np.sum(b_costs))))
        / tot_pro
        * 100.0
    )
    pt_fb = float((np.mean(a_fallbacks) - np.mean(b_fallbacks)) * 100.0)
    return {
        "bootstrap_rounds": rounds,
        "accuracy_diff_pp": _entry(pt_acc, acc_diff),
        "savings_diff_pp": _entry(pt_sav, sav_diff),
        "fallback_diff_pp": _entry(pt_fb, fb_diff),
    }


def compute_platt_vs_comparator_paired(df: pd.DataFrame) -> dict[str, Any]:
    """Recompute exact aligned OOF vectors (N=3,000, same 5 outer folds) for 5a-platt vs comparator."""
    df_sorted = df.sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = (
        df_sorted.drop_duplicates("question_id")
        .sort_values("question_id", kind="stable")
        .reset_index(drop=True)
    )
    gkf = GroupKFold(n_splits=5)

    recs: list[dict[str, Any]] = []
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
        val_costs_piv = inner_val.pivot(index="question_id", columns="model", values="cost")[MODELS]
        val_raw_probs = predict_positive_probabilities(base_pipe, val_prompts)
        val_y = {m: val_scores_piv.loc[val_qids_arr, m].to_numpy(dtype=int) for m in MODELS}

        # Comparator: exact Phase 1 recipe (inner-val tau_uncal + full outer_train refit + outer_train costs)
        tau_comp, _ = _select_tau_from_probs(val_raw_probs, in_costs, val_scores_piv, val_costs_piv, val_qids_arr)
        comp_pipe = make_v21_predictor()
        comp_pipe.fit(outer_train[["prompt", "model"]], outer_train["score"].astype(int))
        out_costs = outer_train.groupby("model")["cost"].mean().to_dict()

        # 5a-platt: OOF Platt on inner_val for tau + full inner_val Platt fit
        val_platt_oof = _cross_fit_calibrate_inner_val(val_raw_probs, val_y, val_qids_arr, "platt")
        tau_platt, _ = _select_tau_from_probs(val_platt_oof, in_costs, val_scores_piv, val_costs_piv, val_qids_arr)
        cal_platt = PerModelCalibrator("platt").fit(val_raw_probs, val_y)

        test_prompts = outer_test.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
        test_qids_arr = test_prompts["question_id"].to_numpy()
        test_scores_piv = outer_test.pivot(index="question_id", columns="model", values="score")[MODELS]
        test_costs_piv = outer_test.pivot(index="question_id", columns="model", values="cost")[MODELS]

        comp_probs = predict_positive_probabilities(comp_pipe, test_prompts)
        sel_comp, fb_comp = route_with_policy(comp_probs, out_costs, tau_comp, FALLBACK_MODEL)

        raw_test_probs = predict_positive_probabilities(base_pipe, test_prompts)
        platt_probs = cal_platt.transform(raw_test_probs)
        sel_platt, fb_platt = route_with_policy(platt_probs, in_costs, tau_platt, FALLBACK_MODEL)

        for i, qid in enumerate(test_qids_arr):
            q = int(qid)
            recs.append(
                {
                    "question_id": q,
                    "pro_cost": float(test_costs_piv.loc[q, FALLBACK_MODEL]),
                    "comp_score": float(test_scores_piv.loc[q, sel_comp[i]]),
                    "comp_cost": float(test_costs_piv.loc[q, sel_comp[i]]),
                    "comp_fb": bool(fb_comp[i]),
                    "platt_score": float(test_scores_piv.loc[q, sel_platt[i]]),
                    "platt_cost": float(test_costs_piv.loc[q, sel_platt[i]]),
                    "platt_fb": bool(fb_platt[i]),
                }
            )

    oof = pd.DataFrame(recs).sort_values("question_id", kind="stable").reset_index(drop=True)
    assert len(oof) == 3000
    return compute_paired_bootstrap_diff(
        oof["platt_score"].to_numpy(),
        oof["platt_cost"].to_numpy(),
        oof["platt_fb"].to_numpy(),
        oof["comp_score"].to_numpy(),
        oof["comp_cost"].to_numpy(),
        oof["comp_fb"].to_numpy(),
        oof["pro_cost"].to_numpy(),
        rounds=10000,
        seed=RANDOM_SEED,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-paired-only",
        action="store_true",
        help="Compute paired bootstrap (5a-platt minus comparator) and update existing phase5_ab.json + phase5_ab.md without re-running MiniLM.",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(DATASET_PATH)
    if args.update_paired_only:
        payload = json.loads(OUTPUT_AB_JSON.read_text(encoding="utf-8"))
        paired_diff = compute_platt_vs_comparator_paired(df)
        payload["platt_vs_comparator_paired_bootstrap"] = paired_diff
        OUTPUT_AB_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(paired_diff, indent=2))
        return 0
    latencies = measure_routing_latencies(df)
    st_model = latencies.pop("st_model")

    # Pre-encode all 3,000 unique prompts once with all-MiniLM-L6-v2
    unique_q = df.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    enc_started = time.perf_counter()
    embs = st_model.encode(unique_q["prompt"].astype(str).tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    encode_all_s = time.perf_counter() - enc_started
    emb_by_qid = {int(q): embs[i] for i, q in enumerate(unique_q["question_id"])}

    res_a = run_attempt_a_calibration(df)
    res_b = run_attempt_b_embeddings(df, emb_by_qid)

    rows = [
        {
            "attempt_id": "comparator",
            "method": COMPARATOR["name"],
            "accuracy_pct": 84.33,
            "retention_pct": COMPARATOR["retention_pct"],
            "savings_pct": COMPARATOR["savings_pct"],
            "fallback_pct": COMPARATOR["fallback_pct"],
            "ni_lb_pp": COMPARATOR["ni_lb_pp"],
            "mcnemar_vs_pro_p": "1.613e-05",
            "mcnemar_vs_flash_p": "1.78e-10",
            "ece_before": round(res_a["calibration"]["base_uncal"]["ece_10bin"], 5),
            "ece_after": "",
            "brier_before": round(res_a["calibration"]["base_uncal"]["brier_score"], 5),
            "brier_after": "",
            "routing_p50_ms": round(latencies["v21_tfidf_lr"]["p50_ms"], 2),
            "routing_p95_ms": round(latencies["v21_tfidf_lr"]["p95_ms"], 2),
            "routing_p99_ms": round(latencies["v21_tfidf_lr"]["p99_ms"], 2),
            "cold_start_ms": round(latencies["v21_tfidf_lr"]["cold_start_ms"], 2),
            "passes_quality_cost_gate": False,
            "passes_latency_gate": latencies["v21_tfidf_lr"]["passes_p95_25ms_gate"],
            "passes_overall_gate": False,
            "notes": "Phase 1 refit comparator baseline",
        },
        _row_for_log(
            "5a-platt",
            res_a["metrics"]["platt"]["name"],
            res_a["metrics"]["platt"],
            res_a["calibration"]["base_uncal"],
            res_a["calibration"]["platt"],
            latencies["a_platt"],
            "Per-model Platt sigmoid fit on inner_val",
        ),
        _row_for_log(
            "5a-isotonic",
            res_a["metrics"]["isotonic"]["name"],
            res_a["metrics"]["isotonic"],
            res_a["calibration"]["base_uncal"],
            res_a["calibration"]["isotonic"],
            latencies["a_isotonic"],
            "Per-model Isotonic regression fit on inner_val",
        ),
        _row_for_log(
            "5b-minilm",
            res_b["metrics"]["minilm_only"]["name"],
            res_b["metrics"]["minilm_only"],
            res_a["calibration"]["base_uncal"],
            res_b["calibration"]["minilm_only"],
            latencies["b_minilm_only"],
            "all-MiniLM-L6-v2 (384-d) + model OHE + LR",
        ),
        _row_for_log(
            "5b-minilm-tfidf",
            res_b["metrics"]["minilm_plus_tfidf"]["name"],
            res_b["metrics"]["minilm_plus_tfidf"],
            res_a["calibration"]["base_uncal"],
            res_b["calibration"]["minilm_plus_tfidf"],
            latencies["b_minilm_plus_tfidf"],
            "all-MiniLM-L6-v2 (384-d) + TF-IDF + model OHE + LR",
        ),
    ]

    ATTEMPT_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ATTEMPT_LOG_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTEMPT_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparator": COMPARATOR,
        "latency_p95_limit_ms": LATENCY_P95_LIMIT_MS,
        "batch_encode_3000_s": float(encode_all_s),
        "latencies": latencies,
        "attempt_a": res_a,
        "attempt_b": res_b,
        "attempt_log_rows": rows,
    }
    OUTPUT_AB_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 5 (Partial: Attempts a & b) — Calibration, Embeddings & Latency",
        "",
        f"- **Generated:** `{payload['timestamp']}`",
        f"- **Comparator (`Refit V2.1 (inner-CV tau)`):** fallback `50.77%`, savings `21.59%`, retention `98.44%`, NI LB `-1.93 pp`.",
        f"- **Latency Gate:** routing-decision `p95 <= 25.0 ms`.",
        "",
        "## Attempt Summary (`results/v5/attempt_log.csv`)",
        "",
        "| ID | Method | Acc | Retention | Savings | Fallback | NI LB | ECE (before→after) | Brier (before→after) | Routing p50/p95/p99 (ms) | Cold Start (ms) | Q/C Gate | Lat Gate | Overall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        ece_str = f"{r['ece_before']}→{r['ece_after']}" if r["ece_after"] != "" else str(r["ece_before"])
        bri_str = f"{r['brier_before']}→{r['brier_after']}" if r["brier_after"] != "" else str(r["brier_before"])
        lines.append(
            f"| {r['attempt_id']} | {r['method']} | {r['accuracy_pct']:.2f}% | {r['retention_pct']:.2f}% | "
            f"{r['savings_pct']:.2f}% | {r['fallback_pct']:.2f}% | {r['ni_lb_pp']:+.2f} pp | "
            f"{ece_str} | {bri_str} | {r['routing_p50_ms']:.2f}/{r['routing_p95_ms']:.2f}/{r['routing_p99_ms']:.2f} | "
            f"{r['cold_start_ms']:.1f} | {r['passes_quality_cost_gate']} | {r['passes_latency_gate']} | {r['passes_overall_gate']} |"
        )
    vc = latencies["v21_mock_concurrency"]
    lines.extend(
        [
            "",
            "## Concurrency Benchmark (Frozen V2.1 `/v1/chat`, Mock Providers)",
            "",
            f"- `N = {vc['n_requests']}` requests at concurrency `{vc['concurrency']}`: wall `{vc['wall_time_s']:.3f}s`, throughput `{vc['throughput_rps']:.1f} req/s`, `p50 = {vc['p50_ms']:.2f} ms`, `p95 = {vc['p95_ms']:.2f} ms`, `p99 = {vc['p99_ms']:.2f} ms`.",
            "",
        ]
    )
    OUTPUT_AB_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {ATTEMPT_LOG_CSV}, {OUTPUT_AB_JSON}, and {OUTPUT_AB_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
