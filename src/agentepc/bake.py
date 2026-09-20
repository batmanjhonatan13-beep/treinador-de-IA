from __future__ import annotations

import json
from pathlib import Path

from agentepc.config import load, resolve
from agentepc.memory import list_knowledge


def _blocked(text: str) -> bool:
    low = text.lower()
    return any(tag in low for tag in (load().get("promote") or {}).get("never_bake") or [])


def facts_from_text(text: str) -> list[str]:
    facts = []
    buf: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            if buf:
                para = " ".join(buf).strip()
                if para and not _blocked(para):
                    facts.append(para[:500])
                buf = []
            continue
        if raw.startswith("- "):
            item = raw[2:].strip()
            if item and not _blocked(item):
                facts.append(item)
            continue
        buf.append(raw)
    if buf:
        para = " ".join(buf).strip()
        if para and not _blocked(para):
            facts.append(para[:500])
    return facts


def selected_files(file: str | None = None) -> list[Path]:
    """Todos os arquivos do caderno, ou so os escolhidos. "a.md + b.md" treina os dois juntos."""
    files = list_knowledge()
    if not file:
        return files
    names = [n.strip() for n in file.split(" + ") if n.strip()]
    picked = [p for p in files if p.name in names]
    missing = set(names) - {p.name for p in picked}
    if missing:
        raise ValueError(f"arquivo nao encontrado no caderno: {', '.join(sorted(missing))}")
    return picked


def facts_from_knowledge(file: str | None = None) -> list[str]:
    facts: list[str] = []
    for path in selected_files(file):
        if path.suffix.lower() == ".jsonl":
            continue
        facts.extend(facts_from_text(path.read_text(encoding="utf-8")))
    # unique preserve order
    seen = set()
    out = []
    for fact in facts:
        if fact not in seen:
            seen.add(fact)
            out.append(fact)
    return out


def jsonl_pairs(file: str | None = None) -> list[dict]:
    """Pares prontos de um .jsonl: {"user","assistant"} ou {"messages":[...]}."""
    rows = []
    for path in selected_files(file):
        if path.suffix.lower() != ".jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            user, asst = item.get("user", ""), item.get("assistant", "")
            if item.get("messages"):
                msgs = item["messages"]
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if user and asst and not _blocked(user + asst):
                rows.append({"user": user, "assistant": asst, "source": path.name})
    return rows


def training_pairs(file: str | None = None) -> list[dict]:
    """Um fato vira varias formas de perguntar a MESMA coisa.

    Regra dura: nunca dois exemplos com o mesmo pergunta e respostas diferentes. Antes
    havia "Me lembra de um fato" apontando para todos os fatos do arquivo; isso ensinava
    o modelo a responder um fato ao acaso em qualquer pergunta.
    """
    from agentepc import exam

    facts = facts_from_knowledge(file)
    out: list[dict] = []
    if not facts:
        return out
    for fact in facts:
        out.append({"user": f"Confirme: {fact}", "assistant": f"Sim. {fact}"})
        parts = exam.split_fact(fact)
        if not parts:
            continue
        stem, ans = parts
        out.append({"user": exam.cloze(stem), "assistant": ans})
        out.append({"user": f"{stem}?", "assistant": ans})
    # uma unica visao geral: pergunta aberta nao pode ter N respostas diferentes
    out.append(
        {
            "user": "O que voce sabe sobre isso?",
            "assistant": "\n".join(f"- {f}" for f in facts[:40]),
        }
    )
    out.extend({"user": r["user"], "assistant": r["assistant"]} for r in jsonl_pairs(file))
    # Conhecimento geral junto: sem isso, dezenas de passadas sobre um punhado de fatos
    # ensinam o modelo a responder o seu arquivo para QUALQUER pergunta. Com isso, o treino
    # tem que preservar o que ele ja sabia enquanto aprende o que e novo.
    from agentepc import geral

    out.extend(geral.pares())
    seen, unique = set(), []
    for row in out:
        if row["user"] in seen:
            continue
        seen.add(row["user"])
        unique.append(row)
    return unique


def to_dataset(file: str | None = None) -> Path:
    """Grava o dataset em disco so para voce poder olhar; quem treina le training_pairs."""
    out = resolve(load()["promote"]["dataset_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in training_pairs(file):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def ready(file: str | None = None) -> tuple[bool, str, dict]:
    facts = facts_from_knowledge(file)
    extra = len(jsonl_pairs(file))
    need = int((load().get("learn") or {}).get("min_facts") or 5)
    total = len(facts) + extra
    stats = {"facts": len(facts), "jsonl": extra, "min_facts": need, "files": len(selected_files(file)), "file": file}
    if total < need:
        return False, f"conhecimento curto ({total}/{need}). Arquivo ja vale; LoRA espera.", stats
    return True, "conhecimento pode ir pro QLoRA. O arquivo continua existindo.", stats
