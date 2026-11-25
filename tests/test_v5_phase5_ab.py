from __future__ import annotations

import numpy as np

from scripts.run_v5_phase5_ab import PerModelCalibrator, compute_ece, compute_paired_bootstrap_diff


def test_compute_ece_perfect_vs_miscalibrated():
    y = np.array([0, 0, 1, 1], dtype=int)
    p_perf = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    p_bad = np.array([0.8, 0.8, 0.2, 0.2], dtype=float)
    assert compute_ece(y, p_perf) == 0.0
    assert compute_ece(y, p_bad) > 0.5


def test_compute_paired_bootstrap_diff_flags_zero_exclusion():
    n = 40
    a_scores = np.ones(n, dtype=float)
    b_scores = np.ones(n, dtype=float)
    pro_costs = np.full(n, 1.0, dtype=float)
    a_costs = np.full(n, 0.5, dtype=float)
    b_costs = np.full(n, 0.9, dtype=float)
    a_fb = np.zeros(n, dtype=bool)
    b_fb = np.ones(n, dtype=bool)
    out = compute_paired_bootstrap_diff(a_scores, a_costs, a_fb, b_scores, b_costs, b_fb, pro_costs, rounds=500, seed=42)
    assert out["accuracy_diff_pp"]["excludes_zero"] is False
    assert out["savings_diff_pp"]["excludes_zero"] is True
    assert out["fallback_diff_pp"]["excludes_zero"] is True


def test_per_model_calibrator_outputs_bounded_probabilities():
    rng = np.random.default_rng(42)
    probs = {
        "deepseek-v3-0324": rng.uniform(0.1, 0.9, size=60),
        "gemini-2.5-flash": rng.uniform(0.1, 0.9, size=60),
        "gemini-2.5-pro": rng.uniform(0.2, 0.95, size=60),
    }
    y = {m: (probs[m] > 0.5).astype(int) for m in probs}
    for method in ("platt", "isotonic"):
        cal = PerModelCalibrator(method).fit(probs, y)
        out = cal.transform(probs)
        for m, arr in out.items():
            assert arr.shape == (60,)
            assert ((arr >= 0.0) & (arr <= 1.0)).all()
