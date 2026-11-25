from __future__ import annotations

import hashlib
import asyncio
import logging
import re

from app.telemetry.models import InferenceTelemetry
from app.telemetry.repository import NoopTelemetryRepository, TelemetryRepository

logger = logging.getLogger(__name__)
MAX_ERROR_MESSAGE_LENGTH = 1000
# A cold connection to local PostgreSQL through Windows/Docker Desktop can
# exceed two seconds while localhost resolves across address families. Keep
# writes bounded, with enough margin for that initial connection handshake.
DEFAULT_TELEMETRY_WRITE_TIMEOUT_S = 5.0


def build_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def sanitize_error_message(
    message: str,
    prompt: str,
    secrets: tuple[str, ...] = (),
) -> str:
    safe = message.replace(prompt, "[PROMPT REDACTED]") if prompt else message
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[REDACTED]")
    safe = re.sub(r"(?i)(x-goog-api-key\s*[:=]\s*)\S+", r"\1[REDACTED]", safe)
    safe = re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+", r"\1[REDACTED]", safe)
    safe = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[REDACTED]", safe)
    return " ".join(safe.split())[:MAX_ERROR_MESSAGE_LENGTH]


class TelemetryService:
    def __init__(
        self,
        repository: TelemetryRepository | None = None,
        write_timeout_s: float = DEFAULT_TELEMETRY_WRITE_TIMEOUT_S,
    ) -> None:
        self.repository = repository or NoopTelemetryRepository()
        self.write_timeout_s = write_timeout_s

    @property
    def is_configured(self) -> bool:
        return not isinstance(self.repository, NoopTelemetryRepository)

    @property
    def is_initialized(self) -> bool:
        return getattr(self.repository, "is_initialized", True)

    @classmethod
    def from_database_url(cls, database_url: str | None) -> "TelemetryService":
        if not database_url or not database_url.strip() or ("<" in database_url and ">" in database_url):
            return cls(NoopTelemetryRepository())
        from app.telemetry.postgres import PostgresTelemetryRepository

        return cls(PostgresTelemetryRepository(database_url))

    async def initialize(self, raise_on_error: bool = False) -> None:
        try:
            await self.repository.initialize()
            if hasattr(self.repository, "migrate"):
                await self.repository.migrate()
        except Exception as exc:
            logger.warning(
                "inference_telemetry_initialization_failed",
                extra={"error_type": type(exc).__name__},
            )
            if raise_on_error:
                raise

    async def record(self, telemetry: InferenceTelemetry) -> None:
        try:
            await asyncio.wait_for(
                self.repository.save(telemetry),
                timeout=self.write_timeout_s,
            )
        except Exception as exc:
            # Database exception messages may include connection details. Log only type.
            logger.warning(
                "inference_telemetry_persistence_failed",
                extra={"error_type": type(exc).__name__, "request_id": telemetry.request_id},
            )

    async def close(self) -> None:
        try:
            await self.repository.close()
        except Exception as exc:
            logger.warning(
                "inference_telemetry_close_failed",
                extra={"error_type": type(exc).__name__},
            )
