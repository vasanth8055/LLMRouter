"""LLMRouter V5 Phase 1 — Leakage-Free 5-Fold Nested CV (Refit V2.1 Recipe).

Eliminates the in-sample contamination of Phase 5's static V2.1 evaluation by
refitting the exact V2.1 TF-IDF + LogisticRegression recipe inside every outer
5-fold GroupKFold(question_id) split, selecting tau via inner 80/20 grouped
validation using the 98%-of-Always-Pro rule over V2.1's threshold grid.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.routing_policy import route_by_quality_and_cost  # noqa: E402

DATASET_PATH = ROOT / "data" / "llmrouter_mmlupro_test3000_long.csv"
FROZEN_V21_PATH = ROOT / "models" / "router_v21_tfidf.joblib"
OUTPUT_JSON = ROOT / "results" / "v5" / "cv_refit_v21.json"
OUTPUT_REPORT = ROOT / "reports" / "phase1_cv_refit.md"

MODELS = ["deepseek-v3-0324", "gemini-2.5-flash", "gemini-2.5-pro"]
FALLBACK_MODEL = "gemini-2.5-pro"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RANDOM_SEED = 42
BOOTSTRAP_ROUNDS = 10000
NON_INFERIORITY_MARGIN_PP = -2.0


def make_v21_predictor() -> Pipeline:
    """Construct the exact V2.1 TF-IDF + one-hot model + Logistic Regression pipeline."""
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


def predict_positive_probabilities(
    predictor: Pipeline, prompts_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Return P(score=1 | prompt, model) for each canonical model aligned to prompts_df."""
    classes = list(predictor.named_steps["classifier"].classes_)
    pos_idx = classes.index(1)
    probs: dict[str, np.ndarray] = {}
    for model in MODELS:
        inp = pd.DataFrame({"prompt": prompts_df["prompt"].values, "model": model})
        probs[model] = predictor.predict_proba(inp)[:, pos_idx]
    return probs


def route_with_policy(
    probs_by_model: dict[str, np.ndarray],
    model_costs: dict[str, float],
    threshold: float,
    fallback_model: str = FALLBACK_MODEL,
) -> tuple[list[str], list[bool]]:
    """Apply production route_by_quality_and_cost to each prompt row."""
    n = len(probs_by_model[fallback_model])
    selected: list[str] = []
    fallbacks: list[bool] = []
    for i in range(n):
        p = {m: float(probs_by_model[m][i]) for m in MODELS}
        decision = route_by_quality_and_cost(p, model_costs, threshold, fallback_model)
        selected.append(str(decision["selected_model"]))
        fallbacks.append(bool(decision["fallback"]))
    return selected, fallbacks


def select_inner_threshold(
    predictor: Pipeline,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    thresholds: list[float] = THRESHOLDS,
) -> tuple[float, dict[str, Any]]:
    """Select tau on inner validation maximizing savings subject to acc >= 98% Always-Pro."""
    train_costs = inner_train_df.groupby("model")["cost"].mean().to_dict()
    val_prompts = (
        inner_val_df.drop_duplicates("question_id")
        .sort_values("question_id")
        .reset_index(drop=True)
    )
    qids = val_prompts["question_id"].to_numpy()
    score_pivot = inner_val_df.pivot(index="question_id", columns="model", values="score")[MODELS]
    cost_pivot = inner_val_df.pivot(index="question_id", columns="model", values="cost")[MODELS]

    val_probs = predict_positive_probabilities(predictor, val_prompts)
    pro_acc = float(score_pivot.loc[qids, FALLBACK_MODEL].mean())
    pro_cost = float(cost_pivot.loc[qids, FALLBACK_MODEL].sum())
    target_acc = 0.98 * pro_acc

    sweep: list[dict[str, Any]] = []
    for tau in thresholds:
        chosen, fbs = route_with_policy(val_probs, train_costs, tau)
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
                "total_cost": tot_cost,
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

    return float(best["threshold"]), {
        "best": best,
        "inner_val_pro_acc": pro_acc,
        "inner_val_target_acc": target_acc,
        "sweep": sweep,
    }


def paired_statistics(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compute exact two-sided McNemar test and 10,000-resample paired bootstrap 95% CIs."""
    n = len(scores_a)
    assert len(scores_b) == n and len(costs_a) == n and len(costs_b) == n

    b = int(np.sum((scores_a == 1) & (scores_b == 0)))
    c = int(np.sum((scores_a == 0) & (scores_b == 1)))
    total_disc = b + c
    p_val = 1.0 if total_disc == 0 else float(binomtest(b, total_disc, p=0.5, alternative="two-sided").pvalue)

    indices = rng.choice(n, size=(BOOTSTRAP_ROUNDS, n), replace=True)
    boot_acc_a = np.mean(scores_a[indices], axis=1)
    boot_acc_b = np.mean(scores_b[indices], axis=1)
    boot_diff = boot_acc_a - boot_acc_b

    boot_cost_a = np.sum(costs_a[indices], axis=1)
    boot_cost_b = np.sum(costs_b[indices], axis=1)
    boot_savings_pct = np.where(boot_cost_b > 0, (boot_cost_b - boot_cost_a) / boot_cost_b * 100.0, 0.0)

    diff_ci = np.percentile(boot_diff, [2.5, 97.5])
    return {
        "mcnemar": {
            "discordant_a_correct_b_wrong": b,
            "discordant_a_wrong_b_correct": c,
            "total_discordant": total_disc,
            "p_value": p_val,
        },
        "bootstrap_95ci": {
            "accuracy_a_ci": [float(x) for x in np.percentile(boot_acc_a, [2.5, 97.5])],
            "accuracy_diff_ci": [float(diff_ci[0]), float(diff_ci[1])],
            "accuracy_diff_pp_ci": [float(diff_ci[0] * 100.0), float(diff_ci[1] * 100.0)],
            "cost_savings_pct_ci": [float(x) for x in np.percentile(boot_savings_pct, [2.5, 97.5])],
        },
    }


def run_nested_cv(
    df: pd.DataFrame,
    frozen_v21_pkg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run strict 5-fold GroupKFold nested CV refitting V2.1 inside each outer fold."""
    df_sorted = df.sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = (
        df_sorted.drop_duplicates("question_id")
        .sort_values("question_id", kind="stable")
        .reset_index(drop=True)
    )
    if len(q_df) != 3000 or len(df_sorted) != 9000:
        raise AssertionError(f"Expected 3000 prompts / 9000 rows, got {len(q_df)} / {len(df_sorted)}")

    gkf = GroupKFold(n_splits=5)
    fold_summaries: list[dict[str, Any]] = []
    oof_records: list[dict[str, Any]] = []

    for fold_idx, (train_q_idx, test_q_idx) in enumerate(gkf.split(q_df, groups=q_df["question_id"]), start=1):
        train_qids = set(map(int, q_df.iloc[train_q_idx]["question_id"]))
        test_qids = set(map(int, q_df.iloc[test_q_idx]["question_id"]))
        assert len(train_qids & test_qids) == 0, f"Leakage in outer fold {fold_idx}"

        outer_train_df = df_sorted[df_sorted["question_id"].isin(train_qids)].copy()
        outer_test_df = df_sorted[df_sorted["question_id"].isin(test_qids)].copy()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED + fold_idx)
        inner_q_df = q_df.iloc[train_q_idx].reset_index(drop=True)
        inner_tr_idx, inner_val_idx = next(gss.split(inner_q_df, groups=inner_q_df["question_id"]))
        inner_tr_qids = set(map(int, inner_q_df.iloc[inner_tr_idx]["question_id"]))
        inner_val_qids = set(map(int, inner_q_df.iloc[inner_val_idx]["question_id"]))

        assert len(inner_tr_qids & inner_val_qids) == 0
        assert len(inner_tr_qids & test_qids) == 0
        assert len(inner_val_qids & test_qids) == 0

        inner_tr_df = outer_train_df[outer_train_df["question_id"].isin(inner_tr_qids)].copy()
        inner_val_df = outer_train_df[outer_train_df["question_id"].isin(inner_val_qids)].copy()

        # Inner fit & threshold selection
        inner_predictor = make_v21_predictor()
        inner_predictor.fit(inner_tr_df[["prompt", "model"]], inner_tr_df["score"].astype(int))
        selected_tau, tau_meta = select_inner_threshold(inner_predictor, inner_tr_df, inner_val_df)

        # Full outer-train refit
        outer_predictor = make_v21_predictor()
        outer_predictor.fit(outer_train_df[["prompt", "model"]], outer_train_df["score"].astype(int))
        outer_train_costs = outer_train_df.groupby("model")["cost"].mean().to_dict()

        test_prompts = (
            outer_test_df.drop_duplicates("question_id")[["question_id", "prompt"]]
            .sort_values("question_id", kind="stable")
            .reset_index(drop=True)
        )
        test_scores = outer_test_df.pivot(index="question_id", columns="model", values="score")[MODELS]
        test_costs = outer_test_df.pivot(index="question_id", columns="model", values="cost")[MODELS]

        test_probs = predict_positive_probabilities(outer_predictor, test_prompts)
        sel_refit_cv, fb_refit_cv = route_with_policy(test_probs, outer_train_costs, selected_tau)
        sel_refit_090, fb_refit_090 = route_with_policy(test_probs, outer_train_costs, 0.90)

        if frozen_v21_pkg is not None:
            frozen_probs = predict_positive_probabilities(frozen_v21_pkg["predictor"], test_prompts)
            frozen_costs = {m: float(v) for m, v in frozen_v21_pkg["training_average_costs"].items()}
            sel_frozen, fb_frozen = route_with_policy(
                frozen_probs, frozen_costs, float(frozen_v21_pkg["selected_threshold"])
            )
        else:
            sel_frozen = [FALLBACK_MODEL] * len(test_prompts)
            fb_frozen = [False] * len(test_prompts)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "outer_train_questions": len(train_qids),
                "inner_train_questions": len(inner_tr_qids),
                "inner_val_questions": len(inner_val_qids),
                "outer_test_questions": len(test_qids),
                "selected_tau": selected_tau,
                "inner_val_accuracy": tau_meta["best"]["accuracy"],
                "inner_val_retention_pct": tau_meta["best"]["retention_pct"],
                "inner_val_savings_pct": tau_meta["best"]["savings_vs_pro_pct"],
                "outer_train_average_costs": {m: float(outer_train_costs[m]) for m in MODELS},
            }
        )

        for i, qid in enumerate(test_prompts["question_id"]):
            qid_int = int(qid)
            oof_records.append(
                {
                    "fold": fold_idx,
                    "question_id": qid_int,
                    "score_deepseek": float(test_scores.loc[qid_int, MODELS[0]]),
                    "cost_deepseek": float(test_costs.loc[qid_int, MODELS[0]]),
                    "score_flash": float(test_scores.loc[qid_int, MODELS[1]]),
                    "cost_flash": float(test_costs.loc[qid_int, MODELS[1]]),
                    "score_pro": float(test_scores.loc[qid_int, MODELS[2]]),
                    "cost_pro": float(test_costs.loc[qid_int, MODELS[2]]),
                    "selected_refit_cv": sel_refit_cv[i],
                    "fallback_refit_cv": fb_refit_cv[i],
                    "score_refit_cv": float(test_scores.loc[qid_int, sel_refit_cv[i]]),
                    "cost_refit_cv": float(test_costs.loc[qid_int, sel_refit_cv[i]]),
                    "selected_refit_090": sel_refit_090[i],
                    "fallback_refit_090": fb_refit_090[i],
                    "score_refit_090": float(test_scores.loc[qid_int, sel_refit_090[i]]),
                    "cost_refit_090": float(test_costs.loc[qid_int, sel_refit_090[i]]),
                    "selected_frozen_insample": sel_frozen[i],
                    "fallback_frozen_insample": fb_frozen[i],
                    "score_frozen_insample": float(test_scores.loc[qid_int, sel_frozen[i]]),
                    "cost_frozen_insample": float(test_costs.loc[qid_int, sel_frozen[i]]),
                }
            )

    oof_df = pd.DataFrame(oof_records).sort_values("question_id", kind="stable").reset_index(drop=True)
    assert len(oof_df) == 3000 and oof_df["question_id"].nunique() == 3000

    # Uniform 3-model random baseline (seed 42) aligned to sorted question_ids
    rand_rng = np.random.default_rng(RANDOM_SEED)
    rand_choices = rand_rng.choice(MODELS, size=len(oof_df)).tolist()
    score_col_map = {
        MODELS[0]: "score_deepseek",
        MODELS[1]: "score_flash",
        MODELS[2]: "score_pro",
    }
    cost_col_map = {
        MODELS[0]: "cost_deepseek",
        MODELS[1]: "cost_flash",
        MODELS[2]: "cost_pro",
    }
    oof_df["selected_random"] = rand_choices
    oof_df["score_random"] = [float(oof_df.loc[i, score_col_map[m]]) for i, m in enumerate(rand_choices)]
    oof_df["cost_random"] = [float(oof_df.loc[i, cost_col_map[m]]) for i, m in enumerate(rand_choices)]

    pro_scores = oof_df["score_pro"].to_numpy()
    pro_costs = oof_df["cost_pro"].to_numpy()
    flash_scores = oof_df["score_flash"].to_numpy()
    flash_costs = oof_df["cost_flash"].to_numpy()

    systems_spec: list[tuple[str, str, np.ndarray, np.ndarray, str | None, str | None, bool]] = [
        ("deepseek", "Always DeepSeek", oof_df["score_deepseek"].to_numpy(), oof_df["cost_deepseek"].to_numpy(), None, None, False),
        ("flash", "Always Gemini Flash", oof_df["score_flash"].to_numpy(), oof_df["cost_flash"].to_numpy(), None, None, False),
        ("pro", "Always Gemini Pro", pro_scores, pro_costs, None, None, False),
        ("random", "Random (3-model, seed 42)", oof_df["score_random"].to_numpy(), oof_df["cost_random"].to_numpy(), "selected_random", None, False),
        ("refit_v21_cv", "Refit V2.1 (inner-CV tau)", oof_df["score_refit_cv"].to_numpy(), oof_df["cost_refit_cv"].to_numpy(), "selected_refit_cv", "fallback_refit_cv", False),
        ("refit_v21_090", "Refit V2.1 (fixed tau=0.90)", oof_df["score_refit_090"].to_numpy(), oof_df["cost_refit_090"].to_numpy(), "selected_refit_090", "fallback_refit_090", False),
    ]
    if frozen_v21_pkg is not None:
        systems_spec.append(
            (
                "frozen_v21_insample",
                "Frozen V2.1 (in-sample, do not cite)",
                oof_df["score_frozen_insample"].to_numpy(),
                oof_df["cost_frozen_insample"].to_numpy(),
                "selected_frozen_insample",
                "fallback_frozen_insample",
                True,
            )
        )

    boot_rng = np.random.default_rng(RANDOM_SEED)
    systems_out: dict[str, Any] = {}
    for key, label, scores, costs, sel_col, fb_col, is_insample in systems_spec:
        acc = float(np.mean(scores))
        pro_acc = float(np.mean(pro_scores))
        tot_cost = float(np.sum(costs))
        mean_cost = float(np.mean(costs))
        pro_tot_cost = float(np.sum(pro_costs))
        retention_pct = float(acc / pro_acc * 100.0)
        savings_pct = float((pro_tot_cost - tot_cost) / pro_tot_cost * 100.0)

        stats_pro = paired_statistics(scores, pro_scores, costs, pro_costs, boot_rng)
        stats_flash = paired_statistics(scores, flash_scores, costs, flash_costs, boot_rng)

        ni_lower_pp = stats_pro["bootstrap_95ci"]["accuracy_diff_pp_ci"][0]
        ni_pass = bool(ni_lower_pp > NON_INFERIORITY_MARGIN_PP)

        dist_info: dict[str, Any] = {}
        if sel_col is not None:
            counts = oof_df[sel_col].value_counts().to_dict()
            dist_info = {
                "counts": {m: int(counts.get(m, 0)) for m in MODELS},
                "percentages": {m: float(counts.get(m, 0) / len(oof_df) * 100.0) for m in MODELS},
            }
            if fb_col is not None:
                dist_info["fallback_count"] = int(oof_df[fb_col].sum())
                dist_info["fallback_rate_pct"] = float(oof_df[fb_col].mean() * 100.0)

        systems_out[key] = {
            "name": label,
            "in_sample_contaminated": is_insample,
            "n_prompts": int(len(oof_df)),
            "accuracy": acc,
            "accuracy_pct": acc * 100.0,
            "accuracy_95ci_pct": [x * 100.0 for x in stats_pro["bootstrap_95ci"]["accuracy_a_ci"]],
            "quality_retention_vs_pro_pct": retention_pct,
            "total_recorded_cost_usd": tot_cost,
            "mean_recorded_cost_usd": mean_cost,
            "cost_savings_vs_pro_pct": savings_pct,
            "cost_savings_vs_pro_95ci_pct": stats_pro["bootstrap_95ci"]["cost_savings_pct_ci"],
            "fallback_rate_pct": dist_info.get("fallback_rate_pct"),
            "routing_distribution": dist_info,
            "vs_always_pro": {
                "diff_pp": float((acc - pro_acc) * 100.0),
                "diff_95ci_pp": stats_pro["bootstrap_95ci"]["accuracy_diff_pp_ci"],
                "mcnemar": stats_pro["mcnemar"],
                "non_inferiority_margin_pp": NON_INFERIORITY_MARGIN_PP,
                "non_inferiority_lower_bound_pp": ni_lower_pp,
                "non_inferiority_passed": ni_pass,
            },
            "vs_always_flash": {
                "diff_pp": float((acc - float(np.mean(flash_scores))) * 100.0),
                "diff_95ci_pp": stats_flash["bootstrap_95ci"]["accuracy_diff_pp_ci"],
                "mcnemar": stats_flash["mcnemar"],
            },
        }

    refit_cv = systems_out["refit_v21_cv"]
    gate_eval = {
        "zero_leakage_verified": True,
        "retention_ge_98pct": refit_cv["quality_retention_vs_pro_pct"] >= 98.0,
        "non_inferiority_gt_minus_2pp": refit_cv["vs_always_pro"]["non_inferiority_passed"],
        "overall_gate_passed": bool(
            refit_cv["quality_retention_vs_pro_pct"] >= 98.0
            and refit_cv["vs_always_pro"]["non_inferiority_passed"]
        ),
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "n_prompts": 3000,
            "outer_cv": "5-fold GroupKFold(question_id)",
            "inner_split": "GroupShuffleSplit(test_size=0.20, random_state=42+fold)",
            "model_recipe": "TF-IDF(ngram_range=(1,2), min_df=2, max_features=30000, sublinear_tf=True) + OneHot(model) + LogisticRegression(C=1.0, max_iter=2000, random_state=42)",
            "threshold_grid": THRESHOLDS,
            "threshold_selection_rule": "max cost_savings_vs_pro subject to inner_val_acc >= 0.98 * inner_val_pro_acc",
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "non_inferiority_margin_pp": NON_INFERIORITY_MARGIN_PP,
        },
        "fold_summaries": fold_summaries,
        "systems": systems_out,
        "gate_evaluation": gate_eval,
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    """Format the Phase 1 report directly from measured JSON metrics."""
    s = result["systems"]
    gate = result["gate_evaluation"]
    folds = result["fold_summaries"]

    lines = [
        "# Phase 1 — Leakage-Free 5-Fold Nested CV (Refit V2.1 Recipe)",
        "",
        f"- **Generated (UTC):** `{result['timestamp_utc']}`",
        f"- **Dataset:** `{result['protocol']['dataset']}` (`N = 3,000` prompts, `9,000` rows)",
        f"- **Protocol:** 5-fold `GroupKFold` by `question_id` (`2,400` outer-train / `600` outer-test); inner 80/20 `GroupShuffleSplit` (`1,920` / `480`) selects $\\tau \\in [0.50, 0.95]$ (step `0.05`) using the $\\ge 98\\%$ Always-Pro retention rule; full outer-train refit before out-of-fold evaluation.",
        f"- **Non-Inferiority Margin:** `{NON_INFERIORITY_MARGIN_PP:.2f} pp` (95% bootstrap CI lower bound of `router - Always Pro` must exceed `-2.00 pp`).",
        "",
        "## 1. Gate Evaluation",
        "",
        f"- **Zero Prompt Leakage Verified:** `{gate['zero_leakage_verified']}`",
        f"- **Refit V2.1 (inner-CV tau) Quality Retention $\\ge 98.0\\%$:** `{gate['retention_ge_98pct']}` (`{s['refit_v21_cv']['quality_retention_vs_pro_pct']:.2f}%`)",
        f"- **Refit V2.1 (inner-CV tau) Non-Inferiority ($> -2.00\\text{{ pp}}$):** `{gate['non_inferiority_gt_minus_2pp']}` (lower bound `{s['refit_v21_cv']['vs_always_pro']['non_inferiority_lower_bound_pp']:+.2f} pp`)",
        f"- **Phase 1 Gate Passed:** **`{'PASS' if gate['overall_gate_passed'] else 'FAIL'}`**",
        "",
        "## 2. Pooled Out-of-Fold Results (`N = 3,000` Prompts)",
        "",
        "| Method | Accuracy (95% CI) | % of Always-Pro | Cost Saving vs Pro (95% CI) | Mean Cost | Fallback Rate | McNemar vs Pro ($p$) | McNemar vs Flash ($p$) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    order = [
        "deepseek",
        "flash",
        "pro",
        "random",
        "refit_v21_cv",
        "refit_v21_090",
        "frozen_v21_insample",
    ]
    for key in order:
        if key not in s:
            continue
        row = s[key]
        ci = row["accuracy_95ci_pct"]
        sci = row["cost_savings_vs_pro_95ci_pct"]
        fb = "N/A" if row["fallback_rate_pct"] is None else f"{row['fallback_rate_pct']:.2f}%"
        p_pro = row["vs_always_pro"]["mcnemar"]["p_value"]
        p_flash = row["vs_always_flash"]["mcnemar"]["p_value"]
        lines.append(
            f"| {row['name']} | {row['accuracy_pct']:.2f}% [{ci[0]:.2f}%, {ci[1]:.2f}%] "
            f"| {row['quality_retention_vs_pro_pct']:.2f}% "
            f"| {row['cost_savings_vs_pro_pct']:.2f}% [{sci[0]:.2f}%, {sci[1]:.2f}%] "
            f"| ${row['mean_recorded_cost_usd']:.6f} | {fb} "
            f"| {p_pro:.3e} | {p_flash:.3e} |"
        )

    lines += [
        "",
        "> **Warning on `Frozen V2.1 (in-sample, do not cite)`:** This column evaluates the static `models/router_v21_tfidf.joblib` artifact across all 3,000 prompts; **2,100 of 3,000 (70%) were in its training split**. It is included solely to expose the in-sample inflation identified in the baseline audit.",
        "",
        "## 3. Non-Inferiority Analysis vs Always Gemini Pro (Margin = `-2.00 pp`)",
        "",
        "| Method | Diff vs Pro (pp) | 95% Bootstrap CI (pp) | CI Lower Bound (pp) | Margin (`-2.00 pp`) | Non-Inferiority |",
        "|---|---:|---:|---:|---:|:---:|",
    ]

    for key in ["refit_v21_cv", "refit_v21_090", "flash", "deepseek", "random", "frozen_v21_insample"]:
        if key not in s:
            continue
        row = s[key]
        vp = row["vs_always_pro"]
        ci = vp["diff_95ci_pp"]
        passed = "PASS" if vp["non_inferiority_passed"] else "FAIL"
        lines.append(
            f"| {row['name']} | {vp['diff_pp']:+.2f} pp | [{ci[0]:+.2f}, {ci[1]:+.2f}] pp "
            f"| {vp['non_inferiority_lower_bound_pp']:+.2f} pp | -2.00 pp | **{passed}** |"
        )

    lines += [
        "",
        "## 4. Per-Fold Inner-CV Threshold Selection & Routing Distribution",
        "",
        "| Outer Fold | Inner Train / Val | Outer Test | Selected $\\tau^*$ | Inner Val Acc | Inner Val Retention | Inner Val Savings |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        lines.append(
            f"| {f['fold']} | {f['inner_train_questions']} / {f['inner_val_questions']} "
            f"| {f['outer_test_questions']} | {f['selected_tau']:.2f} "
            f"| {f['inner_val_accuracy']*100:.2f}% | {f['inner_val_retention_pct']:.2f}% | {f['inner_val_savings_pct']:.2f}% |"
        )

    lines += [
        "",
        "| Method | DeepSeek (%) | Flash (%) | Pro (%) | Fallbacks |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["refit_v21_cv", "refit_v21_090", "frozen_v21_insample"]:
        if key not in s:
            continue
        rd = s[key]["routing_distribution"]
        pcts = rd["percentages"]
        cnts = rd["counts"]
        lines.append(
            f"| {s[key]['name']} "
            f"| {pcts[MODELS[0]]:.2f}% ({cnts[MODELS[0]]}) "
            f"| {pcts[MODELS[1]]:.2f}% ({cnts[MODELS[1]]}) "
            f"| {pcts[MODELS[2]]:.2f}% ({cnts[MODELS[2]]}) "
            f"| {rd.get('fallback_count', 0)} ({rd.get('fallback_rate_pct', 0.0):.2f}%) |"
        )

    lines += [
        "",
        "## 5. Reproduction",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe scripts/run_v5_phase1_cv_refit.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(DATASET_PATH, keep_default_na=False)
    frozen_pkg = joblib.load(FROZEN_V21_PATH) if FROZEN_V21_PATH.exists() else None
    result = run_nested_cv(df, frozen_v21_pkg=frozen_pkg)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    OUTPUT_REPORT.write_text(render_markdown_report(result), encoding="utf-8")

    print(f"Saved {OUTPUT_JSON.relative_to(ROOT)} and {OUTPUT_REPORT.relative_to(ROOT)}")
    for key, row in result["systems"].items():
        vp = row["vs_always_pro"]
        print(
            f"  {row['name']}: acc={row['accuracy_pct']:.2f}% "
            f"ret={row['quality_retention_vs_pro_pct']:.2f}% "
            f"sav={row['cost_savings_vs_pro_pct']:.2f}% "
            f"fb={row['fallback_rate_pct']} "
            f"NI_lb={vp['non_inferiority_lower_bound_pp']:+.2f}pp ({vp['non_inferiority_passed']})"
        )
    print("Gate evaluation:", result["gate_evaluation"])


if __name__ == "__main__":
    main()
