from __future__ import annotations

import json
from pathlib import Path

from agentepc.config import load, resolve
from agentepc.episodes import successes


def _stats(rows: list[dict]) -> dict:
    skills = {row["skill"] for row in rows}
    return {"episodes": len(rows), "skills": len(skills), "skill_names": sorted(skills)}


def ready(rows: list[dict] | None = None) -> tuple[bool, str, dict]:
    cfg = load()["promote"]
    rows = successes() if rows is None else rows
    stats = _stats(rows)
    if stats["episodes"] < cfg["min_episodes"]:
        return (
            False,
            (
                f"poucos sucessos ({stats['episodes']}/{cfg['min_episodes']}). "
                "Isso ainda e caderno, nao peso."
            ),
            stats,
        )
    if stats["skills"] < cfg["min_distinct_skills"]:
        return (
            False,
            (
                f"pouca variedade ({stats['skills']}/{cfg['min_distinct_skills']} skills). "
                "Treinar 30 vezes 'abrir o Chrome' so vicia o modelo."
            ),
            stats,
        )
    return True, "lote passou no filtro; pode gerar dataset e treinar LoRA", stats


def to_dataset() -> Path:
    """Transforma sucessos em pares de instrucao. Ainda nao mexe nos pesos."""
    cfg = load()
    rows = successes()
    out = resolve(cfg["promote"]["dataset_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in rows:
        examples.append(
            {
                "skill": row["skill"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Voce e um agente local. Use o comando que ja funcionou "
                            "nesta maquina. Nao invente path. Se faltar fato de "
                            "maquina, leia data/machine.md."
                        ),
                    },
                    {"role": "user", "content": row["task"]},
                    {
                        "role": "assistant",
                        "content": (
                            f"Skill: {row['skill']}\n"
                            f"Comando:\n```\n{row['command']}\n```\n"
                            f"{row.get('notes') or ''}"
                        ).strip(),
                    },
                ],
            }
        )
    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out
