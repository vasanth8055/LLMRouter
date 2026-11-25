"""Train and package the versioned Platt-calibrated Phase 5 runtime router artifact.

Follows the exact Phase 5 Attempt (5a-platt) protocol on MMLU-Pro test_3000 (N=3,000):
  1. GroupShuffleSplit (80/20, random_state=42) by question_id into 2,400 fit / 600 cal-val prompts.
  2. Fit TF-IDF (1-2gram, min_df=2, max_features=30000, sublinear_tf=True) + model OHE + LR(C=1.0)
     on the 2,400 fit prompts.
  3. Within the 600 cal-val prompts, compute 5-fold GroupKFold OOF Platt-calibrated probabilities to
     select `tau*` over [0.50..0.95] via the 98%-of-Always-Pro retention rule (no tuning on the
     burned 450 test split as a standalone test holdout).
  4. Fit final per-model Platt parameters {model: {coef, intercept}} on all 600 cal-val prompts.
  5. Save self-contained `models/router_v5_platt.joblib` using built-in/sklearn primitives only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_v5_phase1_cv_refit import (  # noqa: E402
    DATASET_PATH,
    FALLBACK_MODEL,
    MODELS,
    RANDOM_SEED,
    make_v21_predictor,
    predict_positive_probabilities,
)
from run_v5_phase5_ab import (  # noqa: E402
    PerModelCalibrator,
    _cross_fit_calibrate_inner_val,
    _select_tau_from_probs,
)

V5_PLATT_MODEL_FILE = ROOT / "models" / "router_v5_platt.joblib"


def train_and_save_v5_platt(output_path: Path = V5_PLATT_MODEL_FILE) -> dict:
    df = pd.read_csv(DATASET_PATH).sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    q_df = df.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    assert len(q_df) == 3000

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    fit_idx, cal_idx = next(gss.split(q_df, groups=q_df["question_id"]))
    fit_qids = set(map(int, q_df.iloc[fit_idx]["question_id"]))
    cal_qids = set(map(int, q_df.iloc[cal_idx]["question_id"]))
    assert fit_qids.isdisjoint(cal_qids)

    fit_df = df[df["question_id"].isin(fit_qids)].copy()
    cal_df = df[df["question_id"].isin(cal_qids)].copy()

    predictor = make_v21_predictor()
    predictor.fit(fit_df[["prompt", "model"]], fit_df["score"].astype(int))
    fit_costs = {m: float(v) for m, v in fit_df.groupby("model")["cost"].mean().to_dict().items()}

    cal_prompts = cal_df.drop_duplicates("question_id").sort_values("question_id", kind="stable").reset_index(drop=True)
    cal_qids_arr = cal_prompts["question_id"].to_numpy()
    cal_scores_piv = cal_df.pivot(index="question_id", columns="model", values="score")[MODELS]
    cal_costs_piv = cal_df.pivot(index="question_id", columns="model", values="cost")[MODELS]
    cal_raw_probs = predict_positive_probabilities(predictor, cal_prompts)
    cal_y = {m: cal_scores_piv.loc[cal_qids_arr, m].to_numpy(dtype=int) for m in MODELS}

    cal_oof = _cross_fit_calibrate_inner_val(cal_raw_probs, cal_y, cal_qids_arr, "platt")
    selected_tau, tau_meta = _select_tau_from_probs(cal_oof, fit_costs, cal_scores_piv, cal_costs_piv, cal_qids_arr)

    calibrator = PerModelCalibrator("platt").fit(cal_raw_probs, cal_y)
    platt_params = {
        m: {
            "coef": float(calibrator._calibrators[m].coef_[0][0]),
            "intercept": float(calibrator._calibrators[m].intercept_[0]),
        }
        for m in MODELS
    }

    package = {
        "predictor": predictor,
        "platt_params": platt_params,
        "selected_threshold": float(selected_tau),
        "training_average_costs": fit_costs,
        "fallback_model": FALLBACK_MODEL,
        "metadata": {
            "version": "v5-platt",
            "features": ["prompt", "model"],
            "calibration": "per_model_platt_inner_val",
            "dataset": str(DATASET_PATH.name),
            "total_prompts": len(q_df),
            "fit_prompts": len(fit_qids),
            "calibration_val_prompts": len(cal_qids),
            "random_seed": RANDOM_SEED,
            "selected_tau": float(selected_tau),
            "cal_val_best": tau_meta["best"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, output_path)
    return package


def main() -> int:
    pkg = train_and_save_v5_platt()
    print(
        json.dumps(
            {
                "artifact": str(V5_PLATT_MODEL_FILE),
                "selected_tau": pkg["selected_threshold"],
                "platt_params": pkg["platt_params"],
                "training_average_costs": pkg["training_average_costs"],
                "cal_val_best": pkg["metadata"]["cal_val_best"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
