from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils import read_params


def main() -> None:
    params = read_params()
    logs_path = Path(params.get("logs", {}).get("moderation_events", "logs/moderation_events.jsonl"))
    output_path = Path(params.get("data", {}).get("output_dir", "data")) / "offline_feature_store" / "offline_training.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not logs_path.exists():
        pd.DataFrame({"event_ts": []}).to_parquet(output_path, index=False)
        print(output_path)
        return

    rows = []
    with open(logs_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "snapshot" in record and isinstance(record["snapshot"], dict):
                rows.append(record["snapshot"])
            else:
                rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame({"event_ts": []})
    frame.to_parquet(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
