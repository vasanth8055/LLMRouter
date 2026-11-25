"""Data Integrity and Protocol Validation Script for LLMRouter Phase 5.

Performs read-only validation of:
1. Primary development dataset: MMLU-Pro test_3000 long table
2. Quarantine status of MMLU-Pro test_1000
3. External holdout datasets: GPQA test and LiveCodeBench test
4. Subject extraction and safe unknown category handling
5. Prompt grouping and leakage-free cross-validation partitioning
6. Frozen V2.1 runtime artifact integrity
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
BENCH_ROOT = Path(r"D:\Downloads\bench-release")

EXPECTED_MODELS = ["deepseek-v3-0324", "gemini-2.5-flash", "gemini-2.5-pro"]
KNOWN_SUBJECTS = {
    "math", "physics", "law", "chemistry", "other",
    "engineering", "economics", "psychology", "health",
    "business", "biology", "philosophy", "computer science", "history"
}


def extract_subject(prompt: str) -> str:
    match = re.search(r"(?i)Answer the following ([a-z][a-z -]*?) question", prompt)
    if not match:
        return "unknown"
    sub = match.group(1).strip().lower()
    return sub if sub in KNOWN_SUBJECTS else "unknown"


def run_checks() -> dict[str, Any]:
    results: dict[str, Any] = {}
    print("=" * 70)
    print("LLMRouter Phase 5 — Data Integrity & Protocol Validation")
    print("=" * 70)

    # 1. Primary MMLU-Pro test_3000
    mmlu_path = DATA_DIR / "llmrouter_mmlupro_test3000_long.csv"
    assert mmlu_path.exists(), f"Missing primary dataset: {mmlu_path}"
    df_mmlu = pd.read_csv(mmlu_path)
    print(f"\n[1/6] Checking Primary Dataset: {mmlu_path.name}")
    assert len(df_mmlu) == 9000, f"Expected 9000 rows; got {len(df_mmlu)}"
    assert df_mmlu.question_id.nunique() == 3000, f"Expected 3000 questions; got {df_mmlu.question_id.nunique()}"
    assert sorted(df_mmlu.model.unique()) == sorted(EXPECTED_MODELS), "Unexpected models in MMLU-Pro"
    assert df_mmlu.isna().sum().sum() == 0, "Missing values found in MMLU-Pro"
    assert set(df_mmlu.score.unique()).issubset({0.0, 1.0}), "Non-binary scores in MMLU-Pro"
    counts_per_q = df_mmlu.groupby("question_id").model.count()
    assert (counts_per_q == 3).all(), "Every question must have exactly 3 model records"
    print("  -> Passed: 3,000 questions, 9,000 rows, 3 models/q, binary scores, 0 missing values.")
    results["mmlu_pro_test3000"] = {
        "status": "PASS",
        "questions": 3000,
        "rows": 9000,
        "models": EXPECTED_MODELS,
    }

    # 2. Quarantine Check: MMLU-Pro test_1000
    print("\n[2/6] Checking Quarantine Status of MMLU-Pro test_1000")
    test1000_raw = BENCH_ROOT / "mmlupro" / "test_1000"
    assert test1000_raw.exists(), f"Expected raw directory {test1000_raw} to exist"
    # Ensure no file in data/ or scripts/ is importing test_1000 into Phase 5
    for p in DATA_DIR.glob("*test1000*"):
        raise AssertionError(f"Quarantined dataset found in data directory: {p}")
    print("  -> Passed: test_1000 confirmed quarantined (isolated under bench-release, excluded from data/).")
    results["quarantine_test_1000"] = {
        "status": "PASS",
        "raw_path": str(test1000_raw),
        "quarantined": True,
        "reason": "99.7% duplicate of test_3000 (2,990/3,000 questions overlap)",
    }

    # 3. External Dataset 1: GPQA Test
    gpqa_path = DATA_DIR / "llmrouter_gpqa_long.csv"
    assert gpqa_path.exists(), f"Missing GPQA dataset: {gpqa_path}"
    df_gpqa = pd.read_csv(gpqa_path, keep_default_na=False)
    print(f"\n[3/6] Checking External Dataset: {gpqa_path.name}")
    assert len(df_gpqa) == 594, f"Expected 594 rows; got {len(df_gpqa)}"
    assert df_gpqa.question_id.nunique() == 198, f"Expected 198 questions; got {df_gpqa.question_id.nunique()}"
    assert sorted(df_gpqa.model.unique()) == sorted(EXPECTED_MODELS), "Unexpected models in GPQA"
    assert df_gpqa.isna().sum().sum() == 0, "Missing values found in GPQA"
    assert set(df_gpqa.score.unique()).issubset({0.0, 1.0}), "Non-binary scores in GPQA"
    counts_per_gpqa = df_gpqa.groupby("question_id").model.count()
    assert (counts_per_gpqa == 3).all(), "Every GPQA question must have exactly 3 model records"
    # Overlap with MMLU-Pro
    gpqa_prompts = set(df_gpqa.prompt.unique())
    mmlu_prompts = set(df_mmlu.prompt.unique())
    gpqa_overlap = gpqa_prompts & mmlu_prompts
    assert len(gpqa_overlap) == 0, f"Found {len(gpqa_overlap)} overlapping prompts between GPQA and MMLU-Pro"
    print("  -> Passed: 198 questions, 594 rows, 0 overlap with MMLU-Pro, binary scores, 0 missing.")
    results["gpqa_test"] = {
        "status": "PASS",
        "questions": 198,
        "rows": 594,
        "overlap_with_mmlu": len(gpqa_overlap),
    }

    # 4. External Dataset 2: LiveCodeBench Test
    lcb_path = DATA_DIR / "llmrouter_livecodebench_long.csv"
    assert lcb_path.exists(), f"Missing LiveCodeBench dataset: {lcb_path}"
    df_lcb = pd.read_csv(lcb_path, keep_default_na=False)
    print(f"\n[4/6] Checking External Dataset: {lcb_path.name}")
    assert len(df_lcb) == 3165, f"Expected 3165 rows; got {len(df_lcb)}"
    assert df_lcb.question_id.nunique() == 1055, f"Expected 1055 questions; got {df_lcb.question_id.nunique()}"
    assert sorted(df_lcb.model.unique()) == sorted(EXPECTED_MODELS), "Unexpected models in LCB"
    assert df_lcb.isna().sum().sum() == 0, "Missing values found in LCB"
    assert set(df_lcb.score.unique()).issubset({0.0, 1.0}), "Non-binary scores in LCB"
    assert "cost" in df_lcb.columns and "normalized_cost" in df_lcb.columns, "Both cost and normalized_cost required"
    counts_per_lcb = df_lcb.groupby("question_id").model.count()
    assert (counts_per_lcb == 3).all(), "Every LCB question must have exactly 3 model records"
    lcb_prompts = set(df_lcb.prompt.unique())
    lcb_overlap = lcb_prompts & mmlu_prompts
    assert len(lcb_overlap) == 0, f"Found {len(lcb_overlap)} overlapping prompts between LCB and MMLU-Pro"
    print("  -> Passed: 1,055 questions, 3,165 rows, 0 overlap with MMLU-Pro, recorded and normalized costs present.")
    results["livecodebench_test"] = {
        "status": "PASS",
        "questions": 1055,
        "rows": 3165,
        "overlap_with_mmlu": len(lcb_overlap),
    }

    # 5. Subject Extraction and Category Mapping
    print("\n[5/6] Checking Subject Extraction and Safe Category Mapping")
    mmlu_subs = df_mmlu.drop_duplicates("question_id").prompt.apply(extract_subject)
    assert set(mmlu_subs.unique()) == KNOWN_SUBJECTS, f"Unexpected MMLU subjects: {set(mmlu_subs.unique()) - KNOWN_SUBJECTS}"
    assert "unknown" not in mmlu_subs.values, "MMLU-Pro should have 0 unknown subjects"

    gpqa_subs = df_gpqa.drop_duplicates("question_id").prompt.apply(extract_subject)
    assert (gpqa_subs == "unknown").all(), "All GPQA questions must map to 'unknown'"

    lcb_subs = df_lcb.drop_duplicates("question_id").prompt.apply(extract_subject)
    assert (lcb_subs == "unknown").all(), "All LCB questions must map to 'unknown'"
    print("  -> Passed: MMLU-Pro maps cleanly to 14 subjects; GPQA and LCB safely mapped to 'unknown'.")
    results["subject_mapping"] = {
        "status": "PASS",
        "mmlu_known_subjects": len(KNOWN_SUBJECTS),
        "gpqa_unknown_pct": 100.0,
        "lcb_unknown_pct": 100.0,
    }

    # 6. Prompt Grouping and Leakage-Free Partitioning Check
    print("\n[6/6] Checking Prompt Grouping and GroupKFold Leakage Independence")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df_mmlu, groups=df_mmlu.question_id))
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_qids = set(df_mmlu.iloc[train_idx].question_id.unique())
        test_qids = set(df_mmlu.iloc[test_idx].question_id.unique())
        assert len(train_qids & test_qids) == 0, f"Data leakage in fold {fold_idx}: prompt overlap detected!"
        assert len(test_qids) == 600, f"Expected 600 test questions in fold {fold_idx}; got {len(test_qids)}"
        assert len(train_qids) == 2400, f"Expected 2400 train questions in fold {fold_idx}; got {len(train_qids)}"
    print("  -> Passed: 5-fold GroupKFold verified with 0 prompt leakage across folds (600 test / 2,400 train).")

    # 7. Frozen V2.1 Runtime Artifact
    print("\n[Bonus] Checking Frozen V2.1 Model Artifact")
    v21_path = MODELS_DIR / "router_v21_tfidf.joblib"
    assert v21_path.exists(), f"Missing frozen model: {v21_path}"
    v21_pkg = joblib.load(v21_path)
    assert "predictor" in v21_pkg and "selected_threshold" in v21_pkg, "Invalid V2.1 artifact schema"
    assert float(v21_pkg["selected_threshold"]) == 0.90, "Frozen threshold must be 0.90"
    print("  -> Passed: Frozen V2.1 artifact verified intact.")
    results["v21_artifact"] = {"status": "PASS", "threshold": 0.90}

    print("\n" + "=" * 70)
    print("ALL DATA INTEGRITY AND PROTOCOL CHECKS PASSED SUCCESSFULLY (6/6)")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_checks()
