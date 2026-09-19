"""Cada extracao e um lote com nome proprio.

Antes a extracao era uma so, presa na memoria do servidor: dar F5 perdia de vista, e
formatar outra coisa atropelava a anterior. Agora cada busca vira um lote em disco, com
metadados, e voce escolhe qual formatar. Depois de formatado o texto bruto e apagado —
fica so o arquivo de fatos.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from agentepc.config import resolve


def pasta() -> Path:
    path = resolve("data/extracoes")
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", texto).strip("-")[:48]


def nome_para(assunto: str, url: str) -> str:
    """Nome do lote: o assunto que voce deu; sem ele, o que a URL revela.

    https://docs.python.org/pt-br/3.14/ -> python-3-14
    https://x.atlassian.net/wiki/spaces/TIME/ -> x-time
    """
    if slug(assunto):
        return slug(assunto)
    p = urllib.parse.urlparse(url)
    host = re.sub(r"^(www|docs|wiki)\.", "", p.netloc.split(":")[0])
    marca = host.split(".")[0]
    # segmentos que dizem alguma coisa: versao, espaco, produto
    partes = [s for s in p.path.split("/") if s and not re.fullmatch(r"[a-z]{2}(-[a-z]{2})?", s)]
    partes = [s for s in partes if s.lower() not in ("wiki", "docs", "spaces", "index.html")]
    return slug("-".join([marca, *partes[-2:]])) or slug(marca) or "extracao"


def caminho_livre(nome: str) -> tuple[str, Path]:
    """Se ja existe um lote com esse nome, acrescenta -2, -3..."""
    base, n = nome, 1
    while (pasta() / f"{base}.md").exists() or (pasta() / f"{base}.json").exists():
        n += 1
        base = f"{nome}-{n}"
    return base, pasta() / f"{base}.md"


def gravar_meta(lote: str, dados: dict) -> None:
    alvo = pasta() / f"{lote}.json"
    atual = ler_meta(lote)
    atual.update(dados)
    atual.setdefault("criado", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    alvo.write_text(json.dumps(atual, ensure_ascii=False, indent=1), encoding="utf-8")


def ler_meta(lote: str) -> dict:
    alvo = pasta() / f"{lote}.json"
    if not alvo.exists():
        return {}
    try:
        return json.loads(alvo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tam(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def listar() -> list[dict]:
    saida = []
    for meta in sorted(pasta().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        lote = meta.stem
        bruto = pasta() / f"{lote}.md"
        fatos = pasta() / f"{lote}-fatos.md"
        dados = ler_meta(lote)
        if not bruto.exists() and not fatos.exists():
            continue
        prog = pasta() / f"{lote}.progresso.json"
        parcial = {}
        if prog.exists():
            try:
                parcial = json.loads(prog.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                parcial = {}
        saida.append({
            **dados,
            "id": lote,
            "progresso": parcial,
            "bruto": str(bruto) if bruto.exists() else "",
            "fatos": str(fatos) if fatos.exists() else "",
            "bytes": _tam(bruto) or _tam(fatos),
            "estado": "formatado" if fatos.exists() else ("pronto" if bruto.exists() else "vazio"),
        })
    return saida


def apagar(lote: str, so_bruto: bool = False) -> dict:
    """Apaga o lote. so_bruto=True tira o texto extraido e mantem os fatos."""
    lote = slug(lote)
    if not lote:
        return {"ok": False, "reason": "lote invalido"}
    alvos = [pasta() / f"{lote}.md", pasta() / f"{lote}.paginas.jsonl"]
    if not so_bruto:
        alvos.append(pasta() / f"{lote}.progresso.json")
    if not so_bruto:
        alvos += [pasta() / f"{lote}-fatos.md", pasta() / f"{lote}.json"]
    achou = False
    for alvo in alvos:
        if alvo.exists():
            alvo.unlink()
            achou = True
    if so_bruto and achou:
        gravar_meta(lote, {"bruto_apagado": True})
    return {"ok": achou, "itens": listar()}
