from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def route_by_quality_and_cost(
    probabilities: Mapping[str, float],
    model_costs: Mapping[str, float],
    threshold: float,
    fallback_model: str,
    actual_costs: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Select the cheapest qualifying model, preserving the unrounded probability.

    ``model_costs`` is the frozen expected/model cost used for selection. When
    supplied, ``actual_costs`` provides the realized cost to report for the
    selected validation observation; it does not affect model selection.
    """
    if fallback_model not in probabilities:
        raise ValueError("Fallback model must be present in probabilities")
    if fallback_model not in model_costs:
        raise ValueError("Fallback model must have a model cost")
    if set(probabilities) - set(model_costs):
        raise ValueError("Every candidate must have a model cost")
    eligible = [model for model, probability in probabilities.items() if float(probability) >= threshold]
    selected = min(eligible, key=lambda model: (float(model_costs[model]), model)) if eligible else fallback_model
    reported_costs = actual_costs if actual_costs is not None else model_costs
    return {
        "selected_model": selected,
        "selected_probability": float(probabilities[selected]),
        "selected_cost": float(reported_costs[selected]),
        "fallback": not bool(eligible),
    }
