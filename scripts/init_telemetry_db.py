"""Create the inference telemetry table without modifying existing rows."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.telemetry.postgres import PostgresTelemetryRepository  # noqa: E402


async def initialize() -> int:
    if not settings.database_url or ("<" in settings.database_url and ">" in settings.database_url):
        print("Set a valid DATABASE_URL in .env without template placeholders before initializing PostgreSQL telemetry.")
        return 1
    repository = PostgresTelemetryRepository(settings.database_url)
    try:
        await repository.initialize()
        await repository.migrate()
    except Exception as exc:
        # Connection exceptions can contain a DSN; report only the exception type.
        print(f"Telemetry schema initialization failed ({type(exc).__name__}).")
        return 1
    finally:
        await repository.close()
    print("inference_requests schema is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(initialize()))
