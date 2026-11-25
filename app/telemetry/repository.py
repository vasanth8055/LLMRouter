from __future__ import annotations

from typing import Protocol

from app.telemetry.models import InferenceTelemetry


class TelemetryRepository(Protocol):
    @property
    def is_initialized(self) -> bool: ...

    async def initialize(self) -> None: ...

    async def save(self, record: InferenceTelemetry) -> None: ...

    async def close(self) -> None: ...


class NoopTelemetryRepository:
    """Used when DATABASE_URL is unset so local/mock inference stays offline."""

    @property
    def is_initialized(self) -> bool:
        return True

    async def initialize(self) -> None:
        return None

    async def save(self, record: InferenceTelemetry) -> None:
        return None

    async def close(self) -> None:
        return None
