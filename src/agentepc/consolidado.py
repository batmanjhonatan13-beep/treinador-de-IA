"""Quando um treino pode ser considerado pronto — a regra muda conforme o tipo.

Texto e classificador tem resposta certa, entao a maquina decide sozinha. Estilo e
personagem nao tem: quem decide e voce, e a sua decisao fica gravada com data. Isso evita
os dois erros: inventar um numero onde nao existe, e deixar tudo "talvez" para sempre.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentepc import classificador, imagem, train
from agentepc.config import resolve

REGRAS = {
    "texto": "3 tentativas seguidas acertando a prova inteira sem consulta — automático",
    "classificador": "acerto na prova (imagens nunca vistas) acima do limiar — automático",
    "imagem": "você aprova as amostras — estilo não tem resposta certa",
}


def _aprovacoes() -> dict:
    arq = resolve("data/aprovacoes.json")
    if arq.exists():
        try:
            return json.loads(arq.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def aprovar(tipo: str, item_id: str, aprovado: bool, nota: str = "") -> dict:
    """Registra o seu julgamento do que a maquina nao consegue medir."""
    dados = _aprovacoes()
    dados[f"{tipo}:{item_id}"] = {
        "aprovado": bool(aprovado), "nota": nota[:200],
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    resolve("data/aprovacoes.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "itens": listar()}


def listar() -> list[dict]:
    """Tudo que foi treinado, com o estado de consolidacao de cada um."""
    aprov = _aprovacoes()
    saida = []

    for r in train.history(100):
        if not r.get("snapshot"):
            continue
        saida.append({
            "tipo": "texto", "id": r["id"], "nome": r.get("file", "?"),
            "consolidado": bool(r.get("consolidated")),
            "detalhe": f"{r.get('passed', 0)}/{r.get('total', 0)} na prova"
                       + (f" · sequência {r.get('streak')}" if r.get("streak") else ""),
            "regra": REGRAS["texto"], "automatico": True,
            "modelo": r.get("base_hf", ""),
        })

    for m in classificador.treinados():
        saida.append({
            "tipo": "classificador", "id": m["id"], "nome": m.get("conjunto", m["id"]),
            "consolidado": bool(m.get("consolidado")),
            "detalhe": f"{m.get('acerto_prova', 0):.0%} em {m.get('imagens_prova', 0)} "
                       f"imagem(ns) nunca vistas · classes: {', '.join(m.get('classes', []))}",
            "regra": REGRAS["classificador"], "automatico": True, "modelo": "resnet18",
        })

    for m in imagem.treinados():
        chave = f"imagem:{m['id']}"
        julgado = aprov.get(chave)
        saida.append({
            "tipo": "imagem", "id": m["id"], "nome": m.get("gatilho") or m["id"],
            "consolidado": bool(julgado and julgado["aprovado"]),
            "reprovado": bool(julgado and not julgado["aprovado"]),
            "detalhe": f"{m.get('tipo', 'estilo')} · {m.get('passos', 0)} passos · "
                       f"{m.get('imagens', 0)} imagem(ns)"
                       + (f" · você aprovou em {julgado['quando'][:10]}" if julgado and julgado["aprovado"]
                          else " · aguardando o seu julgamento" if not julgado else " · você reprovou"),
            "regra": REGRAS["imagem"], "automatico": False,
            "modelo": m.get("base", ""), "amostras": m.get("amostras", []),
        })
    return saida


def consolidados() -> list[dict]:
    return [i for i in listar() if i["consolidado"]]
