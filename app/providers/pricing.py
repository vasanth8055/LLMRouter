from __future__ import annotations


def estimated_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> float | None:
    """Compute a configured live-price estimate, or None if incomplete."""
    if (
        input_tokens is None
        or output_tokens is None
        or input_price_per_million is None
        or output_price_per_million is None
    ):
        return None
    return (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000


def usage_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
