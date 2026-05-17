from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.app.core.database import async_session_maker
from src.app.core.settings import get_settings
from src.app.services.model_registry import ModelRegistryService
from src.utils import read_params


async def main() -> None:
    params = read_params()
    settings = get_settings()
    model_name = params.get("model_registry", {}).get("model_name", settings.model_registry_model_name)

    registry = ModelRegistryService(settings)
    async with async_session_maker() as db:
        row = await registry.rollback_latest(db, model_name=model_name)
        await db.commit()

    report = {
        "model_name": model_name,
        "status": row.status.value if row else "not_found",
        "version": row.version if row else None,
        "traffic_percent": row.traffic_percent if row else 0,
    }
    output = Path("reports/model_rollback_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
