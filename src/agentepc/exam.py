from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from agentepc import bake, ollama
from agentepc.config import load, resolve

# Cada regra pega o pedaco do fato que vira a resposta. A ordem importa: a primeira
# que casar manda. O fallback (ultima palavra) so entra quando nenhuma casa.
_RULES = [
    r"^(?P<stem>.*\bse chamam?)\s+(?P<ans>.+)$",
    r"^(?P<stem>.*\b(?:gosta de|gostam de|adora|prefere|curte|usa|toca|estuda))\s+(?P<ans>.+)$",
    r"^(?P<stem>.*\b(?:trabalha em|trabalha na|trabalha no|mora em|mora na|mora no|nasceu em|nasceu na|nasceu no|fica em|vive em))\s+(?P<ans>.+)$",
    r"^(?P<stem>.*\b(?:e|é|era|foi|sao|são)\s+(?:o|a|os|as|um|uma))\s+(?P<ans>.+)$",
    r"^(?P<stem>.*\b(?:e|é|era|foi|sao|são))\s+(?P<ans>.+)$",
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _refuse(text: str) -> bool:
    n = _norm(text)
    return any(
        p in n
        for p in (
            "nao sei",
            "nao tenho",
            "nao souber",
            "sem inform",
            "nao ha inform",
            "nao foi ensinado",
            "nao conheco",
        )
    )


def _has(haystack: str, needle: str) -> bool:
    """Palavra inteira: 'bia' nao pode passar dentro de 'sabia'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def passed(answer: str, keys: list[str]) -> bool:
    if _refuse(answer):
        return False
    n = _norm(answer)
    return all(_has(n, _norm(k)) for k in keys if k.strip())


def keys_for(answer: str) -> list[str]:
    tokens = _norm(answer).split()
    content = [t for t in tokens if len(t) >= 4 or t.isdigit()][:3]
    return content or tokens[:3]


def split_fact(fact: str) -> tuple[str, str] | None:
    """Separa o fato em (enunciado, resposta). A resposta e o que a prova cobra."""
    f = re.sub(r"\s+", " ", fact.strip()).rstrip(".")
    if len(f) < 12:
        return None
    if ":" in f:
        stem, ans = f.split(":", 1)
    else:
        for rule in _RULES:
            m = re.match(rule, f, re.I)
            if m and len(m.group("ans").split()) <= 8:
                stem, ans = m.group("stem"), m.group("ans")
                break
        else:
            m = re.match(r"(.*)\s(\S+)$", f)
            if not m:
                return None
            stem, ans = m.group(1), m.group(2)
    stem, ans = stem.strip(), ans.strip()
    return (stem, ans) if stem and ans and len(ans) < len(f) else None


def cloze(stem: str) -> str:
    return f"Complete o fato, so a resposta: {stem} ___"


def _phrase(fact: str, stem: str, ans: str) -> str | None:
    """Pede ao Qwem original que redija a pergunta. So aceita a que passa nos filtros."""
    try:
        data = ollama.request(
            "/api/chat",
            {
                "model": ollama.base_name(),
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0},
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Fato: {fact}\n"
                            f"Escreva UMA pergunta curta em portugues cuja resposta seja exatamente: {ans}\n"
                            "Nao inclua a resposta na pergunta. Responda so a pergunta."
                        ),
                    }
                ],
            },
            timeout=120,
        )
        q = ((data.get("message") or {}).get("content") or "").strip().splitlines()[0].strip()
    except Exception:
        return None
    nq = _norm(q)
    # segunda pessoa troca o sujeito do fato ("como voce se chama?")
    if not q.endswith("?") or _norm(ans) in nq or " voce " in f" {nq} ":
        return None
    if not any(t in nq for t in _norm(stem).split() if len(t) >= 4):
        return None
    return q


def _cache_path(file: str | None) -> Path:
    tag = re.sub(r"[^a-z0-9]+", "-", (file or "todos").lower()).strip("-")[:60]
    path = resolve("data/provas") / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fingerprint(file: str | None) -> str:
    h = hashlib.sha1()
    for path in bake.selected_files(file):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build(file: str | None = None, refresh: bool = False) -> list[dict]:
    """Prova do arquivo: uma pergunta por fato, cobrindo todos os temas.

    Fica em cache enquanto o arquivo nao muda, para que duas notas sejam comparaveis
    (e para nao pagar uma chamada ao modelo por fato a cada teste).
    """
    cache, fp = _cache_path(file), _fingerprint(file)
    if not refresh and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fp and saved.get("items"):
            return saved["items"]

    limit = int((load().get("learn") or {}).get("max_questions") or 40)
    seen: set[str] = set()
    out: list[dict] = []
    for path in bake.selected_files(file):
        if path.suffix.lower() == ".jsonl":
            continue
        for fact in bake.facts_from_text(path.read_text(encoding="utf-8")):
            parts = split_fact(fact)
            if not parts:
                continue
            stem, ans = parts
            phrased = _phrase(fact, stem, ans)
            # "frase" e pergunta nova; "cloze" tambem aparece no treino, entao vale menos
            q, kind = (phrased, "frase") if phrased else (cloze(stem), "cloze")
            if _norm(q) in seen:
                continue
            seen.add(_norm(q))
            out.append({"q": q, "keys": keys_for(ans), "kind": kind, "fact": fact, "source": path.name})

    for row in bake.jsonl_pairs(file):
        user, asst = row.get("user", ""), row.get("assistant", "")
        if user and asst and _norm(user) not in seen:
            seen.add(_norm(user))
            out.append({"q": user, "keys": keys_for(asst), "kind": "jsonl", "fact": asst, "source": row.get("source", "")})

    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    cache.write_text(json.dumps({"fingerprint": fp, "items": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def grade(answer: str, item: dict) -> dict:
    return {
        "q": item["q"],
        "keys": item["keys"],
        "kind": item.get("kind", "frase"),
        "answer": answer[:240],
        "ok": passed(answer, item["keys"]),
    }
