from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_v5_phase1_cv_refit import (
    MODELS,
    NON_INFERIORITY_MARGIN_PP,
    THRESHOLDS,
    make_v21_predictor,
    paired_statistics,
    run_nested_cv,
)


def _synthetic_long_df(n_prompts: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    costs = {"deepseek-v3-0324": 0.0005, "gemini-2.5-flash": 0.003, "gemini-2.5-pro": 0.04}
    for qid in range(1, n_prompts + 1):
        easy = qid % 2 == 0
        prompt = (
            f"easy factual arithmetic question {qid} capital sum"
            if easy
            else f"complex multi step formal derivation proof {qid} theorem"
        )
        for model in MODELS:
            if model == "gemini-2.5-pro":
                score = 1 if rng.random() < 0.90 else 0
            elif easy:
                score = 1 if rng.random() < 0.92 else 0
            else:
                score = 1 if rng.random() < 0.50 else 0
            rows.append(
                {
                    "question_id": qid,
                    "prompt": prompt,
                    "ground_truth": "A",
                    "model": model,
                    "score": score,
                    "cost": costs[model],
                    "input_tokens": 100,
                    "output_tokens": 100,
                }
            )
    return pd.DataFrame(rows)


def test_v21_recipe_hyperparameters_match_frozen_spec():
    pipe = make_v21_predictor()
    transformers = {name: trans for name, trans, _ in pipe.named_steps["features"].transformers}
    tfidf = transformers["text"]
    clf = pipe.named_steps["classifier"]
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.min_df == 2
    assert tfidf.max_features == 30000
    assert tfidf.sublinear_tf is True
    assert clf.C == 1.0
    assert clf.max_iter == 2000
    assert clf.random_state == 42
    assert THRESHOLDS == [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def test_paired_statistics_non_inferiority_and_mcnemar():
    rng = np.random.default_rng(42)
    pro = np.array([1] * 85 + [0] * 15)
    router = np.array([1] * 84 + [0] * 16)
    costs_pro = np.full(100, 0.04)
    costs_router = np.full(100, 0.02)
    stats = paired_statistics(router, pro, costs_router, costs_pro, rng)
    assert stats["mcnemar"]["discordant_a_correct_b_wrong"] == 0
    assert stats["mcnemar"]["discordant_a_wrong_b_correct"] == 1
    assert stats["mcnemar"]["p_value"] == 1.0
    assert stats["bootstrap_95ci"]["cost_savings_pct_ci"] == [50.0, 50.0]
    assert NON_INFERIORITY_MARGIN_PP == -2.0


def test_saved_phase1_json_has_no_missing_fields_when_present():
    out = Path("results/v5/cv_refit_v21.json")
    if not out.exists():
        return
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["fold_summaries"]) == 5
    for fold in payload["fold_summaries"]:
        assert fold["outer_train_questions"] == 2400
        assert fold["outer_test_questions"] == 600
        assert fold["inner_train_questions"] + fold["inner_val_questions"] == 2400
    assert "refit_v21_cv" in payload["systems"]
    assert "refit_v21_090" in payload["systems"]
    assert payload["systems"]["refit_v21_cv"]["in_sample_contaminated"] is False
    assert payload["systems"]["frozen_v21_insample"]["in_sample_contaminated"] is True
    assert payload["gate_evaluation"]["zero_leakage_verified"] is True


def test_nested_cv_rejects_non_3000_dataset():
    df = _synthetic_long_df(50)
    try:
        run_nested_cv(df)
    except AssertionError as exc:
        assert "Expected 3000 prompts" in str(exc)
    else:
        raise AssertionError("Expected size guard assertion")
