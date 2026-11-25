from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_v5_phase4_cross_domain import (
    _stratified_group_split,
    infer_domain,
    load_cross_domain_dataset,
    make_cross_domain_predictor,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_CSVS_PRESENT = (
    (ROOT / "data" / "llmrouter_gpqa_long.csv").exists()
    and (ROOT / "data" / "llmrouter_livecodebench_long.csv").exists()
)
SKIP_REASON = (
    "Requires local data/llmrouter_gpqa_long.csv and "
    "data/llmrouter_livecodebench_long.csv (run scripts/build_external_datasets.py)"
)


@pytest.mark.skipif(not EXTERNAL_CSVS_PRESENT, reason=SKIP_REASON)
def test_cross_domain_dataset_has_zero_overlap_and_exact_counts():
    df = load_cross_domain_dataset()
    assert len(df) == 12759
    assert df["group_id"].nunique() == 4253
    assert df.drop_duplicates("group_id")["dataset"].value_counts().to_dict() == {
        "mmlupro": 3000,
        "livecodebench": 1055,
        "gpqa": 198,
    }


@pytest.mark.skipif(not EXTERNAL_CSVS_PRESENT, reason=SKIP_REASON)
def test_infer_domain_uses_prompt_preamble_without_ground_truth():
    df = load_cross_domain_dataset().drop_duplicates("group_id")
    by_ds = df.groupby("dataset")["domain"].value_counts().to_dict()
    assert by_ds == {
        ("gpqa", "gpqa_mcq"): 198,
        ("livecodebench", "code_python"): 1055,
        ("mmlupro", "mmlupro_mcq"): 3000,
    }
    assert infer_domain("Unrelated custom user question?") == "unknown"


@pytest.mark.skipif(not EXTERNAL_CSVS_PRESENT, reason=SKIP_REASON)
def test_stratified_group_split_has_zero_leakage_and_builds_all_variants():
    df = load_cross_domain_dataset()
    groups = df.drop_duplicates("group_id")[["group_id", "dataset"]].head(100).reset_index(drop=True)
    tr_ids, val_ids = _stratified_group_split(groups, test_size=0.20, random_state=42)
    assert tr_ids.isdisjoint(val_ids)
    assert len(tr_ids) + len(val_ids) == 100

    for variant in ("text_only", "with_domain", "with_domain_interaction"):
        pipe = make_cross_domain_predictor(variant)
        assert pipe is not None
