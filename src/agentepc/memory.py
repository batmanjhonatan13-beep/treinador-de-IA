from __future__ import annotations

from pathlib import Path

from agentepc.config import load, resolve


def slug(skill: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in skill.lower())


def _paths() -> tuple[Path, Path]:
    cfg = load()["memory"]
    return resolve(cfg["machine_file"]), resolve(cfg["procedures_dir"])


def write_machine(text: str) -> Path:
    machine, _ = _paths()
    machine.parent.mkdir(parents=True, exist_ok=True)
    machine.write_text(text, encoding="utf-8")
    return machine


def upsert_procedure(skill: str, body: str) -> Path:
    _, folder = _paths()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug(skill)}.md"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def list_procedures() -> list[Path]:
    _, folder = _paths()
    if not folder.exists():
        return []
    return sorted(folder.glob("*.md"))


def profile_path() -> Path:
    path = resolve(load()["memory"]["profile_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_profile() -> str:
    path = profile_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def append_profile(fact: str) -> Path:
    path = profile_path()
    clean = " ".join(fact.split()).strip()
    if not clean:
        raise ValueError("fato vazio")
    prev = read_profile()
    block = f"- {clean}\n"
    if prev:
        path.write_text(prev + "\n" + block, encoding="utf-8")
    else:
        path.write_text("# Sobre o usuario\n\n" + block, encoding="utf-8")
    return path


def knowledge_dir() -> Path:
    path = resolve(load()["memory"]["knowledge_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    raw = Path(name).name
    stem = slug(Path(raw).stem)
    suf = Path(raw).suffix.lower()
    if suf not in {".md", ".txt", ".jsonl"}:
        suf = ".md"
    return f"{stem}{suf}"


def list_knowledge() -> list[Path]:
    files = []
    prof = profile_path()
    if prof.exists():
        files.append(prof)
    files.extend(sorted(p for p in knowledge_dir().iterdir() if p.suffix.lower() in {".md", ".txt", ".jsonl"} and p.is_file()))
    return files


def read_knowledge() -> str:
    chunks = []
    for path in list_knowledge():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(f"## {path.name}\n{text}")
    return "\n\n".join(chunks)


def write_knowledge(name: str, text: str) -> Path:
    clean = text.strip()
    if not clean:
        raise ValueError("arquivo vazio")
    suf = Path(name or "").suffix.lower()
    if suf not in {".md", ".txt", ".jsonl"}:
        raise ValueError("extensao invalida. use .md, .txt ou .jsonl")
    path = knowledge_dir() / _safe_name(name)
    path.write_text(clean + "\n", encoding="utf-8")
    return path


def knowledge_index() -> list[dict]:
    rows = []
    for path in list_knowledge():
        folder = "conhecimento" if path.parent.name == "conhecimento" else path.parent.name
        rows.append({"name": path.name, "folder": folder, "bytes": path.stat().st_size})
    return rows


def save_original(name: str, text: str) -> Path:
    """Copia o arquivo como veio, antes de o modelo ajustar o formato."""
    path = resolve("data/originais")
    path.mkdir(parents=True, exist_ok=True)
    out = path / _safe_name(name)
    out.write_text(text, encoding="utf-8")
    return out


def delete_knowledge(name: str) -> dict:
    """Apaga um arquivo do caderno (e a prova em cache dele). O treino ja feito continua."""
    alvo = knowledge_dir() / _safe_name(name)
    if not alvo.exists():
        prof = profile_path()
        if prof.name != name:
            return {"ok": False, "reason": "arquivo nao encontrado"}
        alvo = prof
    alvo.unlink()
    prova = resolve("data/provas") / (slug(Path(name).stem) + ".json")
    prova.unlink(missing_ok=True)
    return {"ok": True, "file": name, "files": knowledge_index()}
