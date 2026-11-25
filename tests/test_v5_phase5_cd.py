from __future__ import annotations

import pandas as pd

from scripts.run_v5_phase5_cd import extract_subject_and_metadata, make_subject_metadata_predictor


def test_extract_subject_and_metadata_parses_subject_and_falls_back_unknown():
    df = pd.DataFrame(
        [
            {"prompt": "Answer the following physics question. Compute 2+2.", "model": "deepseek-v3-0324"},
            {"prompt": "Write a Python function to sort integers.", "model": "gemini-2.5-flash"},
        ]
    )
    out = extract_subject_and_metadata(df)
    assert out["subject"].tolist() == ["physics", "unknown"]
    assert (out["log_char_len"] > 0).all()
    assert (out["log_word_count"] > 0).all()
    pipe = make_subject_metadata_predictor()
    assert pipe is not None
