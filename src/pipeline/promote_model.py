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
    model_name = settings.model_registry_model_name

    registry = ModelRegistryService(settings)
    async with async_session_maker() as db:
        version = params.get("model_registry", {}).get("promote_version")
        if not version:
            versions = await registry.list_active_versions(db, model_name)
            canary = next((item for item in versions if item.status.value == "canary"), None)
            if canary is None:
                raise ValueError("No canary version is available for promotion.")
            version = canary.version
        row = await registry.promote_version(db, model_name=model_name, version=version)
        await db.commit()

    report = {
        "model_name": row.model_name,
        "version": row.version,
        "status": row.status.value,
        "traffic_percent": row.traffic_percent,
    }
    output = Path("reports/model_promotion_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
