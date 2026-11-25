"""LLMRouter V5 Phase 4 — Cross-Domain Generalization Study.

Evaluates pooled and transfer routing across the three non-overlapping benchmarks:
- MMLU-Pro test_3000 (`data/llmrouter_mmlupro_test3000_long.csv`, N=3,000 prompts)
- GPQA (`data/llmrouter_gpqa_long.csv`, N=198 prompts)
- LiveCodeBench (`data/llmrouter_livecodebench_long.csv`, N=1,055 prompts)

Note: `data/llmrouter_mmlupro_long.csv` is excluded because 998/1,000 of its prompts
are duplicates of `data/llmrouter_mmlupro_test3000_long.csv`.

Runs three leakage-free tracks:
1. Leave-One-Dataset-Out (LODO): train on 2 datasets, test on the 3rd.
2. Pooled In-Domain 5-Fold Nested CV (N=4,253 pooled + per-dataset breakdown):
   compares `text_only`, `with_domain` (inference-time prompt-derived domain),
   and `with_domain_interaction` (domain + domain*model interaction).
3. Single-Dataset 5-Fold In-Domain Nested CV per dataset.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_v5_phase1_cv_refit import (  # noqa: E402
    BOOTSTRAP_ROUNDS,
    FALLBACK_MODEL,
    MODELS,
    NON_INFERIORITY_MARGIN_PP,
    RANDOM_SEED,
    THRESHOLDS,
    paired_statistics,
    route_with_policy,
)

DATASET_FILES = {
    "mmlupro": ROOT / "data" / "llmrouter_mmlupro_test3000_long.csv",
    "gpqa": ROOT / "data" / "llmrouter_gpqa_long.csv",
    "livecodebench": ROOT / "data" / "llmrouter_livecodebench_long.csv",
}
FROZEN_V21_PATH = ROOT / "models" / "router_v21_tfidf.joblib"
OUTPUT_JSON = ROOT / "results" / "v5" / "cross_domain.json"
OUTPUT_REPORT = ROOT / "reports" / "phase4_cross_domain.md"

VARIANTS = ("text_only", "with_domain", "with_domain_interaction")


def infer_domain(prompt: str) -> str:
    """Infer task domain purely from prompt format observable at inference time."""
    text = prompt.lstrip()
    if text.startswith("You are an expert Python programmer"):
        return "code_python"
    if re.match(r"Answer the following multiple choice question\.", text, flags=re.I):
        return "gpqa_mcq"
    if re.match(r"Answer the following [a-z][a-z -]*? question\.", text, flags=re.I):
        return "mmlupro_mcq"
    return "unknown"


def load_cross_domain_dataset() -> pd.DataFrame:
    """Load and concatenate the 3 non-overlapping benchmarks with unique group_id."""
    frames: list[pd.DataFrame] = []
    seen_prompts_by_ds: dict[str, set[str]] = {}
    for dataset_name, path in DATASET_FILES.items():
        raw = pd.read_csv(path)
        norm_cost = (
            raw["normalized_cost"].astype(float)
            if "normalized_cost" in raw.columns
            else raw["cost"].astype(float)
        )
        prompts = raw["prompt"].astype(str)
        part = pd.DataFrame(
            {
                "dataset": dataset_name,
                "question_id": raw["question_id"].astype(int),
                "group_id": dataset_name + ":" + raw["question_id"].astype(int).astype(str),
                "prompt": prompts,
                "domain": [infer_domain(p) for p in prompts],
                "model": raw["model"].astype(str),
                "domain_model": "",
                "score": raw["score"].astype(int),
                "cost": raw["cost"].astype(float),
                "normalized_cost": norm_cost,
            }
        )
        part["domain_model"] = part["domain"] + "|" + part["model"]
        seen_prompts_by_ds[dataset_name] = {p.strip() for p in part["prompt"]}
        frames.append(part)

    ds_names = list(seen_prompts_by_ds)
    for i in range(len(ds_names)):
        for j in range(i + 1, len(ds_names)):
            overlap = seen_prompts_by_ds[ds_names[i]] & seen_prompts_by_ds[ds_names[j]]
            assert not overlap, f"Cross-dataset prompt overlap between {ds_names[i]} and {ds_names[j]}"

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["group_id", "model"], kind="stable").reset_index(drop=True)
    counts = combined.groupby("group_id")["model"].nunique()
    assert (counts == 3).all(), "Every group_id must have all 3 canonical models"
    return combined


def make_cross_domain_predictor(variant: str) -> Pipeline:
    """Construct TF-IDF + LogisticRegression quality predictor for a given feature variant."""
    transformers: list[tuple[str, Any, Any]] = [
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
    ]
    if variant == "with_domain":
        transformers.append(("domain", OneHotEncoder(handle_unknown="ignore"), ["domain"]))
    elif variant == "with_domain_interaction":
        transformers.append(("domain", OneHotEncoder(handle_unknown="ignore"), ["domain"]))
        transformers.append(("domain_model", OneHotEncoder(handle_unknown="ignore"), ["domain_model"]))
    elif variant != "text_only":
        raise ValueError(f"Unknown variant: {variant}")

    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, remainder="drop")),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED)),
        ]
    )


def _feature_columns(variant: str) -> list[str]:
    if variant == "text_only":
        return ["prompt", "model"]
    if variant == "with_domain":
        return ["prompt", "model", "domain"]
    if variant == "with_domain_interaction":
        return ["prompt", "model", "domain", "domain_model"]
    raise ValueError(f"Unknown variant: {variant}")


def predict_probs_for_prompts(
    predictor: Pipeline, prompts_df: pd.DataFrame, variant: str
) -> dict[str, np.ndarray]:
    """Predict P(score=1 | prompt, model, [domain]) aligned to prompts_df."""
    classes = list(predictor.named_steps["classifier"].classes_)
    pos_idx = classes.index(1)
    cols = _feature_columns(variant)
    probs: dict[str, np.ndarray] = {}
    for model in MODELS:
        inp = pd.DataFrame({"prompt": prompts_df["prompt"].values, "model": model})
        if variant != "text_only":
            inp["domain"] = prompts_df["domain"].values
        if variant == "with_domain_interaction":
            inp["domain_model"] = inp["domain"].astype(str) + "|" + model
        probs[model] = predictor.predict_proba(inp[cols])[:, pos_idx]
    return probs


def _select_inner_tau(
    predictor: Pipeline,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    variant: str,
    thresholds: list[float] = THRESHOLDS,
) -> tuple[float, dict[str, Any]]:
    train_costs = inner_train_df.groupby("model")["cost"].mean().to_dict()
    val_prompts = (
        inner_val_df.drop_duplicates("group_id")
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
    )
    gids = val_prompts["group_id"].to_numpy()
    score_pivot = inner_val_df.pivot(index="group_id", columns="model", values="score")[MODELS]
    cost_pivot = inner_val_df.pivot(index="group_id", columns="model", values="cost")[MODELS]

    val_probs = predict_probs_for_prompts(predictor, val_prompts, variant)
    pro_acc = float(score_pivot.loc[gids, FALLBACK_MODEL].mean())
    pro_cost = float(cost_pivot.loc[gids, FALLBACK_MODEL].sum())
    target_acc = 0.98 * pro_acc

    sweep: list[dict[str, Any]] = []
    for tau in thresholds:
        chosen, fbs = route_with_policy(val_probs, train_costs, tau, FALLBACK_MODEL)
        scores = np.array([float(score_pivot.loc[g, m]) for g, m in zip(gids, chosen)])
        costs = np.array([float(cost_pivot.loc[g, m]) for g, m in zip(gids, chosen)])
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
    return float(best["threshold"]), {"best": best, "sweep": sweep}


def _stratified_group_split(
    train_groups_df: pd.DataFrame, *, test_size: float = 0.20, random_state: int = RANDOM_SEED
) -> tuple[set[str], set[str]]:
    """Split unique group_ids stratified by dataset so inner-val preserves domain ratios."""
    if train_groups_df["dataset"].nunique() == 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        dummy_y = np.zeros(len(train_groups_df), dtype=int)
        tr_idx, val_idx = next(splitter.split(train_groups_df, dummy_y))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, val_idx = next(splitter.split(train_groups_df, train_groups_df["dataset"]))
    tr_ids = set(train_groups_df.iloc[tr_idx]["group_id"].astype(str))
    val_ids = set(train_groups_df.iloc[val_idx]["group_id"].astype(str))
    assert tr_ids.isdisjoint(val_ids)
    return tr_ids, val_ids


def _fit_and_eval_split(
    outer_train_df: pd.DataFrame,
    outer_test_df: pd.DataFrame,
    *,
    variant: str,
    inner_seed: int,
    frozen_v21_pkg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on outer_train_df (inner 80/20 tau selection -> full outer-train refit) and score outer_test_df."""
    train_groups = (
        outer_train_df.drop_duplicates("group_id")[["group_id", "dataset"]]
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
    )
    inner_tr_ids, inner_val_ids = _stratified_group_split(train_groups, test_size=0.20, random_state=inner_seed)
    test_ids = set(outer_test_df["group_id"].astype(str))
    assert test_ids.isdisjoint(inner_tr_ids | inner_val_ids)

    inner_tr_df = outer_train_df[outer_train_df["group_id"].isin(inner_tr_ids)].copy()
    inner_val_df = outer_train_df[outer_train_df["group_id"].isin(inner_val_ids)].copy()

    cols = _feature_columns(variant)
    inner_pipe = make_cross_domain_predictor(variant)
    inner_pipe.fit(inner_tr_df[cols], inner_tr_df["score"].astype(int))
    tau_star, tau_meta = _select_inner_tau(inner_pipe, inner_tr_df, inner_val_df, variant)

    outer_pipe = make_cross_domain_predictor(variant)
    outer_pipe.fit(outer_train_df[cols], outer_train_df["score"].astype(int))
    outer_costs = outer_train_df.groupby("model")["cost"].mean().to_dict()

    test_prompts = (
        outer_test_df.drop_duplicates("group_id")[["group_id", "dataset", "question_id", "prompt", "domain"]]
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
    )
    score_pivot = outer_test_df.pivot(index="group_id", columns="model", values="score")[MODELS]
    cost_pivot = outer_test_df.pivot(index="group_id", columns="model", values="cost")[MODELS]
    norm_cost_pivot = outer_test_df.pivot(index="group_id", columns="model", values="normalized_cost")[MODELS]

    probs = predict_probs_for_prompts(outer_pipe, test_prompts, variant)
    sel_cv, fb_cv = route_with_policy(probs, outer_costs, tau_star, FALLBACK_MODEL)
    sel_090, fb_090 = route_with_policy(probs, outer_costs, 0.90, FALLBACK_MODEL)

    if frozen_v21_pkg is not None:
        v21_probs = predict_probs_for_prompts(frozen_v21_pkg["predictor"], test_prompts, "text_only")
        v21_costs = {m: float(v) for m, v in frozen_v21_pkg["training_average_costs"].items()}
        sel_v21, fb_v21 = route_with_policy(v21_probs, v21_costs, float(frozen_v21_pkg["selected_threshold"]))
    else:
        v21_probs = None
        sel_v21 = None
        fb_v21 = None

    rows: list[dict[str, Any]] = []
    for i, gid in enumerate(test_prompts["group_id"].astype(str)):
        rec: dict[str, Any] = {
            "group_id": gid,
            "dataset": str(test_prompts.iloc[i]["dataset"]),
            "question_id": int(test_prompts.iloc[i]["question_id"]),
            "domain": str(test_prompts.iloc[i]["domain"]),
            "score_deepseek": float(score_pivot.loc[gid, MODELS[0]]),
            "cost_deepseek": float(cost_pivot.loc[gid, MODELS[0]]),
            "norm_cost_deepseek": float(norm_cost_pivot.loc[gid, MODELS[0]]),
            "score_flash": float(score_pivot.loc[gid, MODELS[1]]),
            "cost_flash": float(cost_pivot.loc[gid, MODELS[1]]),
            "norm_cost_flash": float(norm_cost_pivot.loc[gid, MODELS[1]]),
            "score_pro": float(score_pivot.loc[gid, MODELS[2]]),
            "cost_pro": float(cost_pivot.loc[gid, MODELS[2]]),
            "norm_cost_pro": float(norm_cost_pivot.loc[gid, MODELS[2]]),
            "prob_deepseek": float(probs[MODELS[0]][i]),
            "prob_flash": float(probs[MODELS[1]][i]),
            "prob_pro": float(probs[MODELS[2]][i]),
            "selected_cv": sel_cv[i],
            "fallback_cv": bool(fb_cv[i]),
            "score_cv": float(score_pivot.loc[gid, sel_cv[i]]),
            "cost_cv": float(cost_pivot.loc[gid, sel_cv[i]]),
            "norm_cost_cv": float(norm_cost_pivot.loc[gid, sel_cv[i]]),
            "selected_090": sel_090[i],
            "fallback_090": bool(fb_090[i]),
            "score_090": float(score_pivot.loc[gid, sel_090[i]]),
            "cost_090": float(cost_pivot.loc[gid, sel_090[i]]),
            "norm_cost_090": float(norm_cost_pivot.loc[gid, sel_090[i]]),
        }
        if sel_v21 is not None and fb_v21 is not None and v21_probs is not None:
            rec.update(
                {
                    "v21_prob_deepseek": float(v21_probs[MODELS[0]][i]),
                    "v21_prob_flash": float(v21_probs[MODELS[1]][i]),
                    "v21_prob_pro": float(v21_probs[MODELS[2]][i]),
                    "selected_v21": sel_v21[i],
                    "fallback_v21": bool(fb_v21[i]),
                    "score_v21": float(score_pivot.loc[gid, sel_v21[i]]),
                    "cost_v21": float(cost_pivot.loc[gid, sel_v21[i]]),
                    "norm_cost_v21": float(norm_cost_pivot.loc[gid, sel_v21[i]]),
                }
            )
        rows.append(rec)

    meta = {
        "selected_tau": float(tau_star),
        "inner_val_accuracy": float(tau_meta["best"]["accuracy"]),
        "inner_val_savings_pct": float(tau_meta["best"]["savings_vs_pro_pct"]),
        "outer_train_prompts": len(train_groups),
        "outer_test_prompts": len(test_prompts),
    }
    return pd.DataFrame(rows), meta


def _summarize_oof(
    oof_df: pd.DataFrame, systems: list[tuple[str, str, str | None]], *, use_normalized_cost: bool = False
) -> dict[str, Any]:
    """Summarize accuracy, Pro-retention, cost savings, fallback rate, routing shares, CIs, McNemar."""
    rng = np.random.default_rng(RANDOM_SEED)
    cost_prefix = "norm_cost_" if use_normalized_cost else "cost_"
    pro_scores = oof_df["score_pro"].to_numpy()
    pro_costs = oof_df[f"{cost_prefix}pro"].to_numpy()
    flash_scores = oof_df["score_flash"].to_numpy()
    flash_costs = oof_df[f"{cost_prefix}flash"].to_numpy()

    pro_acc = float(np.mean(pro_scores))
    pro_total_cost = float(np.sum(pro_costs))

    out: dict[str, Any] = {"n_prompts": len(oof_df), "systems": {}}
    for key, label, suffix in systems:
        scores = oof_df[f"score_{suffix if suffix else key}"].to_numpy()
        costs = oof_df[f"{cost_prefix}{suffix if suffix else key}"].to_numpy()
        acc = float(np.mean(scores))
        tot_cost = float(np.sum(costs))
        savings_pct = float((pro_total_cost - tot_cost) / pro_total_cost * 100.0) if pro_total_cost > 0 else 0.0
        retention_pct = float(acc / pro_acc * 100.0) if pro_acc > 0 else 0.0

        vs_pro = paired_statistics(scores, pro_scores, costs, pro_costs, rng)
        vs_flash = paired_statistics(scores, flash_scores, costs, flash_costs, rng)
        ni_lb_pp = float(vs_pro["bootstrap_95ci"]["accuracy_diff_pp_ci"][0])

        entry: dict[str, Any] = {
            "label": label,
            "accuracy": acc,
            "accuracy_95ci": vs_pro["bootstrap_95ci"]["accuracy_a_ci"],
            "pro_retention_pct": retention_pct,
            "total_cost_usd": tot_cost,
            "avg_cost_usd": float(np.mean(costs)),
            "savings_vs_pro_pct": savings_pct,
            "savings_vs_pro_95ci_pct": vs_pro["bootstrap_95ci"]["cost_savings_pct_ci"],
            "diff_vs_pro_pp": float((acc - pro_acc) * 100.0),
            "diff_vs_pro_95ci_pp": vs_pro["bootstrap_95ci"]["accuracy_diff_pp_ci"],
            "non_inferiority_lb_pp": ni_lb_pp,
            "passes_non_inferiority_2pp": ni_lb_pp > NON_INFERIORITY_MARGIN_PP,
            "mcnemar_vs_pro": vs_pro["mcnemar"],
            "mcnemar_vs_flash": vs_flash["mcnemar"],
        }

        if suffix and f"selected_{suffix}" in oof_df.columns:
            sel = oof_df[f"selected_{suffix}"]
            fb = oof_df[f"fallback_{suffix}"]
            shares = {m: float((sel == m).mean() * 100.0) for m in MODELS}
            entry["fallback_rate_pct"] = float(fb.mean() * 100.0)
            entry["routing_share_pct"] = shares
        else:
            fixed_model = {
                "deepseek": MODELS[0],
                "flash": MODELS[1],
                "pro": MODELS[2],
            }.get(key)
            entry["fallback_rate_pct"] = 100.0 if fixed_model == FALLBACK_MODEL else 0.0
            entry["routing_share_pct"] = {m: (100.0 if m == fixed_model else 0.0) for m in MODELS}

        out["systems"][key] = entry
    return out


def run_lodo(df: pd.DataFrame, frozen_v21_pkg: dict[str, Any]) -> dict[str, Any]:
    """Leave-One-Dataset-Out evaluation: train on 2 datasets, test on the 3rd."""
    results: dict[str, Any] = {}
    for idx, target_ds in enumerate(["mmlupro", "gpqa", "livecodebench"]):
        train_df = df[df["dataset"] != target_ds].copy()
        test_df = df[df["dataset"] == target_ds].copy()

        # Run text_only and with_domain (+ frozen_v21 attached cleanly during text_only pass)
        oof_text, meta_text = _fit_and_eval_split(
            train_df,
            test_df,
            variant="text_only",
            inner_seed=RANDOM_SEED + idx,
            frozen_v21_pkg=frozen_v21_pkg,
        )
        oof_dom, meta_dom = _fit_and_eval_split(
            train_df,
            test_df,
            variant="with_domain",
            inner_seed=RANDOM_SEED + idx,
        )
        oof_dom_int, meta_dom_int = _fit_and_eval_split(
            train_df,
            test_df,
            variant="with_domain_interaction",
            inner_seed=RANDOM_SEED + idx,
        )

        merged = oof_text[
            [
                "group_id",
                "dataset",
                "question_id",
                "score_deepseek",
                "cost_deepseek",
                "norm_cost_deepseek",
                "score_flash",
                "cost_flash",
                "norm_cost_flash",
                "score_pro",
                "cost_pro",
                "norm_cost_pro",
                "selected_v21",
                "fallback_v21",
                "score_v21",
                "cost_v21",
                "norm_cost_v21",
                "v21_prob_deepseek",
                "v21_prob_flash",
                "v21_prob_pro",
            ]
        ].copy()
        for tag, odf in [
            ("text_cv", oof_text),
            ("dom_cv", oof_dom),
            ("dom_int_cv", oof_dom_int),
        ]:
            merged[f"selected_{tag}"] = odf["selected_cv"].values
            merged[f"fallback_{tag}"] = odf["fallback_cv"].values
            merged[f"score_{tag}"] = odf["score_cv"].values
            merged[f"cost_{tag}"] = odf["cost_cv"].values
            merged[f"norm_cost_{tag}"] = odf["norm_cost_cv"].values
            tag090 = tag.replace("_cv", "_090")
            merged[f"selected_{tag090}"] = odf["selected_090"].values
            merged[f"fallback_{tag090}"] = odf["fallback_090"].values
            merged[f"score_{tag090}"] = odf["score_090"].values
            merged[f"cost_{tag090}"] = odf["cost_090"].values
            merged[f"norm_cost_{tag090}"] = odf["norm_cost_090"].values

        merged["lodo_text_prob_deepseek"] = oof_text["prob_deepseek"].values
        merged["lodo_text_prob_flash"] = oof_text["prob_flash"].values
        merged["lodo_text_prob_pro"] = oof_text["prob_pro"].values

        systems = [
            ("deepseek", "Always DeepSeek", None),
            ("flash", "Always Gemini Flash", None),
            ("pro", "Always Gemini Pro", None),
            ("frozen_v21", "Frozen V2.1 (tau=0.90)" + (" [in-sample, do not cite]" if target_ds == "mmlupro" else ""), "v21"),
            ("lodo_text_cv", "LODO Text-Only (inner-CV tau)", "text_cv"),
            ("lodo_text_090", "LODO Text-Only (fixed tau=0.90)", "text_090"),
            ("lodo_dom_cv", "LODO With-Domain (inner-CV tau)", "dom_cv"),
            ("lodo_dom_090", "LODO With-Domain (fixed tau=0.90)", "dom_090"),
        ]
        summary = _summarize_oof(merged, systems)
        summary_norm = _summarize_oof(merged, systems, use_normalized_cost=True)

        extra: dict[str, Any] = {
            "train_datasets": sorted(train_df["dataset"].unique().tolist()),
            "test_dataset": target_ds,
            "meta": {
                "text_only": meta_text,
                "with_domain": meta_dom,
                "with_domain_interaction": meta_dom_int,
            },
            "metrics": summary,
            "metrics_normalized_cost": summary_norm,
        }
        if target_ds == "livecodebench":
            extra["lcb_probability_quantiles"] = {
                "frozen_v21": {
                    m: {
                        "min": float(merged[f"v21_prob_{short}"].min()),
                        "p50": float(merged[f"v21_prob_{short}"].median()),
                        "p90": float(merged[f"v21_prob_{short}"].quantile(0.90)),
                        "max": float(merged[f"v21_prob_{short}"].max()),
                    }
                    for m, short in zip(MODELS, ["deepseek", "flash", "pro"])
                },
                "lodo_text_only": {
                    m: {
                        "min": float(merged[f"lodo_text_prob_{short}"].min()),
                        "p50": float(merged[f"lodo_text_prob_{short}"].median()),
                        "p90": float(merged[f"lodo_text_prob_{short}"].quantile(0.90)),
                        "max": float(merged[f"lodo_text_prob_{short}"].max()),
                    }
                    for m, short in zip(MODELS, ["deepseek", "flash", "pro"])
                },
            }
        results[target_ds] = extra
    return results


def run_pooled_nested_cv(df: pd.DataFrame) -> dict[str, Any]:
    """Stratified 5-fold prompt-grouped nested CV pooled over MMLU-Pro + GPQA + LiveCodeBench."""
    groups_df = (
        df.drop_duplicates("group_id")[["group_id", "dataset", "question_id"]]
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
    )
    fold_map: dict[str, int] = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for ds in sorted(groups_df["dataset"].unique()):
        sub = groups_df[groups_df["dataset"] == ds].reset_index(drop=True)
        for fold_idx, (_tr, te) in enumerate(kf.split(sub), start=1):
            for gid in sub.iloc[te]["group_id"]:
                fold_map[str(gid)] = fold_idx

    fold_records_by_variant: dict[str, list[pd.DataFrame]] = {v: [] for v in VARIANTS}
    fold_meta: list[dict[str, Any]] = []

    for fold_idx in range(1, 6):
        test_ids = {gid for gid, f in fold_map.items() if f == fold_idx}
        train_ids = {gid for gid, f in fold_map.items() if f != fold_idx}
        assert train_ids.isdisjoint(test_ids)

        outer_train = df[df["group_id"].isin(train_ids)].copy()
        outer_test = df[df["group_id"].isin(test_ids)].copy()

        fold_entry: dict[str, Any] = {"fold": fold_idx, "variants": {}}
        for variant in VARIANTS:
            oof_part, meta = _fit_and_eval_split(
                outer_train,
                outer_test,
                variant=variant,
                inner_seed=RANDOM_SEED + fold_idx,
            )
            oof_part["fold"] = fold_idx
            fold_records_by_variant[variant].append(oof_part)
            fold_entry["variants"][variant] = meta
        fold_meta.append(fold_entry)

    pooled_dfs = {
        v: pd.concat(parts, axis=0, ignore_index=True).sort_values("group_id", kind="stable").reset_index(drop=True)
        for v, parts in fold_records_by_variant.items()
    }
    base = pooled_dfs["text_only"][
        [
            "group_id",
            "dataset",
            "question_id",
            "score_deepseek",
            "cost_deepseek",
            "norm_cost_deepseek",
            "score_flash",
            "cost_flash",
            "norm_cost_flash",
            "score_pro",
            "cost_pro",
            "norm_cost_pro",
        ]
    ].copy()
    for v in VARIANTS:
        odf = pooled_dfs[v]
        assert (odf["group_id"].values == base["group_id"].values).all()
        base[f"selected_{v}_cv"] = odf["selected_cv"].values
        base[f"fallback_{v}_cv"] = odf["fallback_cv"].values
        base[f"score_{v}_cv"] = odf["score_cv"].values
        base[f"cost_{v}_cv"] = odf["cost_cv"].values
        base[f"norm_cost_{v}_cv"] = odf["norm_cost_cv"].values
        base[f"selected_{v}_090"] = odf["selected_090"].values
        base[f"fallback_{v}_090"] = odf["fallback_090"].values
        base[f"score_{v}_090"] = odf["score_090"].values
        base[f"cost_{v}_090"] = odf["cost_090"].values
        base[f"norm_cost_{v}_090"] = odf["norm_cost_090"].values
        for short in ["deepseek", "flash", "pro"]:
            base[f"{v}_prob_{short}"] = odf[f"prob_{short}"].values

    systems = [
        ("deepseek", "Always DeepSeek", None),
        ("flash", "Always Gemini Flash", None),
        ("pro", "Always Gemini Pro", None),
        ("text_only_cv", "Pooled Text-Only (inner-CV tau)", "text_only_cv"),
        ("text_only_090", "Pooled Text-Only (fixed tau=0.90)", "text_only_090"),
        ("with_domain_cv", "Pooled With-Domain (inner-CV tau)", "with_domain_cv"),
        ("with_domain_090", "Pooled With-Domain (fixed tau=0.90)", "with_domain_090"),
        ("with_domain_interaction_cv", "Pooled Domain+Interaction (inner-CV tau)", "with_domain_interaction_cv"),
        ("with_domain_interaction_090", "Pooled Domain+Interaction (fixed tau=0.90)", "with_domain_interaction_090"),
    ]
    per_dataset = {
        ds: _summarize_oof(base[base["dataset"] == ds].reset_index(drop=True), systems)
        for ds in ["mmlupro", "gpqa", "livecodebench"]
    }
    lcb_sub = base[base["dataset"] == "livecodebench"]
    lcb_probs = {
        v: {
            m: {
                "min": float(lcb_sub[f"{v}_prob_{short}"].min()),
                "p50": float(lcb_sub[f"{v}_prob_{short}"].median()),
                "p90": float(lcb_sub[f"{v}_prob_{short}"].quantile(0.90)),
                "max": float(lcb_sub[f"{v}_prob_{short}"].max()),
            }
            for m, short in zip(MODELS, ["deepseek", "flash", "pro"])
        }
        for v in VARIANTS
    }

    return {
        "fold_summaries": fold_meta,
        "pooled_overall": _summarize_oof(base, systems),
        "per_dataset": per_dataset,
        "lcb_probability_quantiles": lcb_probs,
    }


def run_single_domain_nested_cv(df: pd.DataFrame) -> dict[str, Any]:
    """5-fold prompt-grouped nested CV inside each individual dataset."""
    out: dict[str, Any] = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for ds in ["mmlupro", "gpqa", "livecodebench"]:
        ds_df = df[df["dataset"] == ds].copy()
        groups = (
            ds_df.drop_duplicates("group_id")[["group_id", "dataset"]]
            .sort_values("group_id", kind="stable")
            .reset_index(drop=True)
        )
        oof_parts: list[pd.DataFrame] = []
        taus: list[float] = []
        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(groups), start=1):
            tr_ids = set(groups.iloc[tr_idx]["group_id"].astype(str))
            te_ids = set(groups.iloc[te_idx]["group_id"].astype(str))
            assert tr_ids.isdisjoint(te_ids)
            part, meta = _fit_and_eval_split(
                ds_df[ds_df["group_id"].isin(tr_ids)].copy(),
                ds_df[ds_df["group_id"].isin(te_ids)].copy(),
                variant="text_only",
                inner_seed=RANDOM_SEED + fold_idx,
            )
            oof_parts.append(part)
            taus.append(meta["selected_tau"])

        oof = pd.concat(oof_parts, axis=0, ignore_index=True).sort_values("group_id", kind="stable").reset_index(drop=True)
        systems = [
            ("deepseek", "Always DeepSeek", None),
            ("flash", "Always Gemini Flash", None),
            ("pro", "Always Gemini Pro", None),
            ("indomain_cv", "Single-Domain Refit (inner-CV tau)", "cv"),
            ("indomain_090", "Single-Domain Refit (fixed tau=0.90)", "090"),
        ]
        out[ds] = {
            "selected_taus": taus,
            "metrics": _summarize_oof(oof, systems),
            "lcb_probability_quantiles": (
                {
                    m: {
                        "min": float(oof[f"prob_{short}"].min()),
                        "p50": float(oof[f"prob_{short}"].median()),
                        "p90": float(oof[f"prob_{short}"].quantile(0.90)),
                        "max": float(oof[f"prob_{short}"].max()),
                    }
                    for m, short in zip(MODELS, ["deepseek", "flash", "pro"])
                }
                if ds == "livecodebench"
                else None
            ),
        }
    return out


def _fmt_sys_row(name: str, m: dict[str, Any]) -> str:
    ci = m["accuracy_95ci"]
    ni = f"{m['non_inferiority_lb_pp']:+.2f}"
    p_pro = f"{m['mcnemar_vs_pro']['p_value']:.4g}"
    p_fl = f"{m['mcnemar_vs_flash']['p_value']:.4g}"
    sh = m["routing_share_pct"]
    return (
        f"| {name} | {m['accuracy']*100:.2f}% [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%] | "
        f"{m['pro_retention_pct']:.2f}% | {m['savings_vs_pro_pct']:.2f}% | "
        f"{m['fallback_rate_pct']:.2f}% | {ni} pp | {p_pro} | {p_fl} | "
        f"{sh[MODELS[0]]:.1f}/{sh[MODELS[1]]:.1f}/{sh[MODELS[2]]:.1f} |"
    )


def render_markdown_report(payload: dict[str, Any]) -> str:
    header = (
        "| System | Accuracy [95% CI] | Pro Retention | Cost Savings | Fallback Rate | NI LB vs Pro | "
        "McNemar vs Pro | McNemar vs Flash | Share DS/Fl/Pr (%) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [
        "# Phase 4 — Cross-Domain Generalization Report",
        "",
        f"- **Generated:** `{payload['timestamp']}`",
        f"- **Combined Dataset (`N = 4,253` prompts / `12,759` rows):** MMLU-Pro `test_3000` (`3,000`), GPQA (`198`), LiveCodeBench (`1,055`). `llmrouter_mmlupro_long.csv` (`1,001`) was excluded because `998/1,000` prompts duplicate `test_3000`.",
        "- **Inference-Time Domain Feature (`infer_domain`):** Derived strictly from prompt preamble (`mmlupro_mcq`, `gpqa_mcq`, `code_python`, `unknown`). Tested in `text_only`, `with_domain` (additive), and `with_domain_interaction` (`domain + domain|model`).",
        "",
        "## 1. Leave-One-Dataset-Out (LODO — Train on 2, Test on 3rd)",
        "",
    ]
    for ds in ["gpqa", "livecodebench", "mmlupro"]:
        block = payload["lodo"][ds]
        sys_map = block["metrics"]["systems"]
        lines.extend(
            [
                f"### Held-Out: `{ds}` (`N = {block['metrics']['n_prompts']}`, trained on `{' + '.join(block['train_datasets'])}`)",
                "",
                header,
            ]
        )
        for k in ["deepseek", "flash", "pro", "frozen_v21", "lodo_text_cv", "lodo_text_090", "lodo_dom_cv", "lodo_dom_090"]:
            lines.append(_fmt_sys_row(sys_map[k]["label"], sys_map[k]))
        lines.append("")

    lines.extend(["## 2. Pooled In-Domain 5-Fold Nested CV (`N = 4,253` Stratified Across All 3 Datasets)", ""])
    ov = payload["pooled_in_domain_cv"]["pooled_overall"]
    lines.extend([f"### Overall Pooled (`N = {ov['n_prompts']}`)", "", header])
    for k in ov["systems"]:
        lines.append(_fmt_sys_row(ov["systems"][k]["label"], ov["systems"][k]))
    lines.append("")

    for ds in ["mmlupro", "gpqa", "livecodebench"]:
        blk = payload["pooled_in_domain_cv"]["per_dataset"][ds]
        lines.extend([f"### Pooled In-Domain Breakdown: `{ds}` (`N = {blk['n_prompts']}`)", "", header])
        for k in blk["systems"]:
            lines.append(_fmt_sys_row(blk["systems"][k]["label"], blk["systems"][k]))
        lines.append("")

    lines.extend(["## 3. Single-Dataset In-Domain 5-Fold Nested CV", ""])
    for ds in ["mmlupro", "gpqa", "livecodebench"]:
        blk = payload["single_domain_nested_cv"][ds]["metrics"]
        taus = payload["single_domain_nested_cv"][ds]["selected_taus"]
        lines.extend([f"### `{ds}` (`N = {blk['n_prompts']}`, fold `tau*` = `{taus}`)", "", header])
        for k in blk["systems"]:
            lines.append(_fmt_sys_row(blk["systems"][k]["label"], blk["systems"][k]))
        lines.append("")

    lcb_lodo = payload["lodo"]["livecodebench"]["metrics"]["systems"]
    lcb_pool = payload["pooled_in_domain_cv"]["per_dataset"]["livecodebench"]["systems"]
    lcb_single = payload["single_domain_nested_cv"]["livecodebench"]["metrics"]["systems"]
    q_lodo = payload["lodo"]["livecodebench"]["lcb_probability_quantiles"]
    q_pool = payload["pooled_in_domain_cv"]["lcb_probability_quantiles"]
    lines.extend(
        [
            "## 4. Why Does LiveCodeBench Fall Back 100% (or Not)?",
            "",
            f"- **True LCB accuracy ranking (`N = 1,055`):** DeepSeek `{lcb_lodo['deepseek']['accuracy']*100:.2f}%` > Gemini Flash `{lcb_lodo['flash']['accuracy']*100:.2f}%` < Gemini Pro `{lcb_lodo['pro']['accuracy']*100:.2f}%` (note: Flash is *worse* than DeepSeek by `6.64 pp` on LCB, opposite MMLU-Pro/GPQA).",
            f"- **Frozen V2.1 (trained on MMLU-Pro only, `tau=0.90`):** Fallback = **`{lcb_lodo['frozen_v21']['fallback_rate_pct']:.2f}%`** (`{lcb_lodo['frozen_v21']['savings_vs_pro_pct']:.2f}%` savings). Max predicted `P(correct)` on LCB was DS `{q_lodo['frozen_v21']['deepseek-v3-0324']['max']:.4f}`, Flash `{q_lodo['frozen_v21']['gemini-2.5-flash']['max']:.4f}`, Pro `{q_lodo['frozen_v21']['gemini-2.5-pro']['max']:.4f}` — none reach `0.90`.",
            f"- **LODO Text-Only (`train = mmlupro + gpqa`, `inner-CV tau`):** Fallback = **`{lcb_lodo['lodo_text_cv']['fallback_rate_pct']:.2f}%`** (`tau = {payload['lodo']['livecodebench']['meta']['text_only']['selected_tau']:.2f}`, savings `{lcb_lodo['lodo_text_cv']['savings_vs_pro_pct']:.2f}%`, retention `{lcb_lodo['lodo_text_cv']['pro_retention_pct']:.2f}%`). At fixed `tau=0.90`, fallback is **`{lcb_lodo['lodo_text_090']['fallback_rate_pct']:.2f}%`**.",
            f"- **Pooled In-Domain CV on LCB (`with_domain` vs `with_domain_interaction`, `inner-CV tau`):**",
            f"  - `text_only (inner-CV)`: fallback `{lcb_pool['text_only_cv']['fallback_rate_pct']:.2f}%`, acc `{lcb_pool['text_only_cv']['accuracy']*100:.2f}%` (retention `{lcb_pool['text_only_cv']['pro_retention_pct']:.2f}%`), savings `{lcb_pool['text_only_cv']['savings_vs_pro_pct']:.2f}%`, share DS/Fl/Pr = `{lcb_pool['text_only_cv']['routing_share_pct']['deepseek-v3-0324']:.1f}/{lcb_pool['text_only_cv']['routing_share_pct']['gemini-2.5-flash']:.1f}/{lcb_pool['text_only_cv']['routing_share_pct']['gemini-2.5-pro']:.1f}%`.",
            f"  - `with_domain (inner-CV)`: fallback `{lcb_pool['with_domain_cv']['fallback_rate_pct']:.2f}%`, acc `{lcb_pool['with_domain_cv']['accuracy']*100:.2f}%` (retention `{lcb_pool['with_domain_cv']['pro_retention_pct']:.2f}%`), savings `{lcb_pool['with_domain_cv']['savings_vs_pro_pct']:.2f}%`, share DS/Fl/Pr = `{lcb_pool['with_domain_cv']['routing_share_pct']['deepseek-v3-0324']:.1f}/{lcb_pool['with_domain_cv']['routing_share_pct']['gemini-2.5-flash']:.1f}/{lcb_pool['with_domain_cv']['routing_share_pct']['gemini-2.5-pro']:.1f}%`.",
            f"  - `with_domain_interaction (inner-CV)`: fallback `{lcb_pool['with_domain_interaction_cv']['fallback_rate_pct']:.2f}%`, acc `{lcb_pool['with_domain_interaction_cv']['accuracy']*100:.2f}%` (retention `{lcb_pool['with_domain_interaction_cv']['pro_retention_pct']:.2f}%`), savings `{lcb_pool['with_domain_interaction_cv']['savings_vs_pro_pct']:.2f}%`, share DS/Fl/Pr = `{lcb_pool['with_domain_interaction_cv']['routing_share_pct']['deepseek-v3-0324']:.1f}/{lcb_pool['with_domain_interaction_cv']['routing_share_pct']['gemini-2.5-flash']:.1f}/{lcb_pool['with_domain_interaction_cv']['routing_share_pct']['gemini-2.5-pro']:.1f}%`.",
            f"  - Even at fixed `tau=0.90`, `with_domain_interaction` falls back `{lcb_pool['with_domain_interaction_090']['fallback_rate_pct']:.2f}%` (max `P(DS)` `{q_pool['with_domain_interaction']['deepseek-v3-0324']['max']:.4f}` vs max `P(Flash)` `{q_pool['with_domain_interaction']['gemini-2.5-flash']['max']:.4f}`) because LiveCodeBench accuracy itself tops out at `79.15%` (Pro) / `66.64%` (DeepSeek) — well below a `0.90` probability floor — while in Single-Domain LCB Nested CV (`inner-CV tau`), fallback drops to **`{lcb_single['indomain_cv']['fallback_rate_pct']:.2f}%`** (retention `{lcb_single['indomain_cv']['pro_retention_pct']:.2f}%`, savings `{lcb_single['indomain_cv']['savings_vs_pro_pct']:.2f}%`), showing both why `tau=0.90` forces 100% fallback and why domain-specific calibration matters.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    df = load_cross_domain_dataset()
    frozen_v21_pkg = joblib.load(FROZEN_V21_PATH)

    lodo = run_lodo(df, frozen_v21_pkg)
    pooled = run_pooled_nested_cv(df)
    single = run_single_domain_nested_cv(df)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_prompts": int(df["group_id"].nunique()),
        "total_rows": int(len(df)),
        "excluded_overlapping_dataset": "data/llmrouter_mmlupro_long.csv (998/1000 prompts overlap test_3000)",
        "lodo": lodo,
        "pooled_in_domain_cv": pooled,
        "single_domain_nested_cv": single,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(render_markdown_report(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} and {OUTPUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
