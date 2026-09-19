from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentepc.config import load, resolve


def episodes_path() -> Path:
    path = resolve(load()["promote"]["episodes_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append(
    *,
    skill: str,
    task: str,
    command: str,
    ok: bool,
    notes: str = "",
    source: str = "human",
) -> dict:
    """Grava um episodio. Sucesso != peso. E so um fato para o caderno/dataset."""
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "skill": skill,
        "task": task,
        "command": command,
        "ok": ok,
        "notes": notes,
        "source": source,
        "baked": False,
    }
    path = episodes_path()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_all() -> list[dict]:
    path = episodes_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def successes() -> list[dict]:
    return [row for row in read_all() if row.get("ok")]
