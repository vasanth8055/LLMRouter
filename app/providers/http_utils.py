from __future__ import annotations

import asyncio
from contextvars import ContextVar
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

T = TypeVar("T")

MAX_PROVIDER_ERROR_DETAIL_CHARS = 400
REQUEST_DEADLINE_AT: ContextVar[float | None] = ContextVar(
    "llmrouter_request_deadline_at", default=None
)
REQUEST_RETRY_COUNT: ContextVar[list[int] | None] = ContextVar(
    "llmrouter_request_retry_count", default=None
)


def _safe_error_detail(
    response: httpx.Response,
    sensitive_values: tuple[str, ...] = (),
) -> str | None:
    """Extract only a bounded provider error message and redact credentials."""
    try:
        payload = response.json()
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        detail = error.get("message") if isinstance(error, dict) else None
    except (ValueError, TypeError):
        return None
    if not isinstance(detail, str) or not detail.strip():
        return None

    secrets: set[str] = set()
    try:
        request = response.request
    except RuntimeError:
        request = None
    if request is not None:
        for header in ("x-goog-api-key", "authorization"):
            value = request.headers.get(header)
            if value:
                secrets.add(value)
        query_key = request.url.params.get("key")
        if query_key:
            secrets.add(query_key)
    secrets.update(value for value in sensitive_values if value)
    for secret in secrets:
        detail = detail.replace(secret, "[REDACTED]")
    detail = re.sub(
        r"(?i)(x-goog-api-key\s*[:=]\s*)\S+",
        r"\1[REDACTED]",
        detail,
    )
    detail = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        detail,
    )
    detail = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[REDACTED]", detail)
    detail = " ".join(detail.split())
    return detail[:MAX_PROVIDER_ERROR_DETAIL_CHARS]


def raise_for_provider_status(
    response: httpx.Response,
    provider_name: str,
    *,
    safe_context: str | None = None,
    sensitive_values: tuple[str, ...] = (),
) -> None:
    status = response.status_code
    if status < 400:
        return
    # Do not include response bodies, request URLs, or headers: providers can
    # echo user content, and URLs/headers may contain credential material.
    message = f"{provider_name} API request failed with HTTP {status}."
    if status in (401, 403):
        raise ProviderAuthenticationError(
            f"{provider_name} authentication failed (HTTP {status}); check the configured API key.",
            status_code=status,
        )
    if status == 429:
        raise ProviderRateLimitError(
            f"{provider_name} rate limit reached (HTTP 429).",
            retryable=True,
            status_code=status,
        )
    retryable = status >= 500
    if (status == 404 or status >= 500) and provider_name == "Gemini" and safe_context:
        detail = _safe_error_detail(response, sensitive_values)
        message = f"Gemini API request to {safe_context} failed with HTTP {status}."
        if detail:
            message += f" Provider message: {detail}"
        raise ProviderError(message, retryable=retryable, status_code=status)
    raise ProviderError(message, retryable=retryable, status_code=status)


def translate_request_error(exc: httpx.RequestError, provider_name: str) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        from app.providers.errors import ProviderTimeoutError

        return ProviderTimeoutError(f"{provider_name} request timed out.")
    return ProviderError(
        f"Could not complete {provider_name} API request ({type(exc).__name__}).",
        retryable=False,
    )


async def async_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay_s: float,
    max_delay_s: float,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand_fn: Callable[[], float] = random.random,
    deadline_at: float | None = None,
    time_fn: Callable[[], float] = time.monotonic,
) -> T:
    """Retry transient provider errors (429/5xx/timeouts) with exponential backoff + jitter."""
    effective_deadline = REQUEST_DEADLINE_AT.get() if deadline_at is None else deadline_at
    retry_tracker = REQUEST_RETRY_COUNT.get()
    attempt = 0
    while True:
        if effective_deadline is not None:
            remaining = effective_deadline - time_fn()
            if remaining <= 0:
                raise ProviderTimeoutError("Request exceeded overall deadline.")
        try:
            if effective_deadline is None:
                return await operation()
            try:
                return await asyncio.wait_for(
                    operation(), timeout=max(0.001, effective_deadline - time_fn())
                )
            except asyncio.TimeoutError as timeout_exc:
                raise ProviderTimeoutError("Request exceeded overall deadline.") from timeout_exc
        except ProviderError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            raw_delay = min(max_delay_s, base_delay_s * (2 ** attempt))
            jittered_delay = raw_delay * (0.5 + 0.5 * rand_fn())
            if effective_deadline is not None:
                remaining = effective_deadline - time_fn()
                if remaining <= 0 or jittered_delay >= remaining:
                    raise ProviderTimeoutError(
                        "Request exceeded overall deadline before retry."
                    ) from exc
            if retry_tracker is not None:
                retry_tracker[0] += 1
            attempt += 1
            if jittered_delay > 0:
                await sleep_fn(jittered_delay)
