"""Junta imagens para treinar um estilo, com a licenca de cada uma anotada.

Busca em fontes que dizem a licenca: Openverse (indexa Flickr, museus, Wikimedia) e
Wikimedia Commons. Nao raspa Pinterest, Instagram nem sites de arte: la a imagem quase
sempre tem dono e uso restrito, e o arquivo baixado nao diz nada sobre isso.

Cada imagem baixada vem acompanhada do autor, da licenca e do endereco de origem, gravados
em creditos.json. Se depois voce publicar algo feito com esse treino, e nesse arquivo que
esta o que precisa ser citado.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from agentepc.config import resolve
from agentepc.lotes import slug

UA = {"User-Agent": "agente-pc/1.0 (treinador de IA local; https://github.com/batmanjhonatan13-beep/treinador-de-IA)"}
PAUSA = 0.7   # respiro entre pedidos: Openverse e Wikimedia cortam quem chega em rajada


def _token() -> str:
    """Chave da Openverse, se voce tiver uma.

    Sem cadastro a Openverse libera poucas buscas por hora — e quando corta, responde 401.
    Com uma chave (gratuita, em api.openverse.org/v1/auth_tokens/register) o teto sobe
    muito. Vem de OPENVERSE_TOKEN no ambiente ou de `openverse_token:` no config.yaml.
    """
    import os

    from agentepc.config import load

    return (os.environ.get("OPENVERSE_TOKEN")
            or str(load().get("openverse_token") or "")).strip()


def abrir(url: str, timeout: int = 60, tentativas: int = 3) -> bytes:
    """Busca uma URL com pausa e segunda chance.

    Sem isso a coleta morre no meio a toa: o Wikimedia responde 429 ("devagar") quando
    varios arquivos saem juntos, e a Openverse responde 401 para quem nao tem cadastro
    depois de algumas buscas seguidas. Nos dois casos basta esperar um pouco.
    """
    erro: Exception = RuntimeError("sem tentativa")
    for n in range(max(1, tentativas)):
        time.sleep(PAUSA * (n + 1))
        cabecalho = dict(UA)
        if "openverse.org" in url and _token():
            cabecalho["Authorization"] = f"Bearer {_token()}"
        try:
            pedido = urllib.request.Request(url, headers=cabecalho)
            with urllib.request.urlopen(pedido, timeout=timeout) as r:
                return r.read(60_000_000)
        except urllib.error.HTTPError as exc:
            erro = exc
            if exc.code not in (401, 429, 500, 502, 503):
                raise
        except Exception as exc:      # rede instavel tambem merece segunda chance
            erro = exc
    raise erro


def json_de(url: str, timeout: int = 45) -> dict:
    return json.loads(abrir(url, timeout=timeout))


def _humano(exc: Exception) -> str:
    codigo = getattr(exc, "code", None)
    if codigo == 401:
        return ("a Openverse corta quem busca muito sem cadastro (401). Espere alguns minutos, "
                "use o Wikimedia junto, ou cadastre uma chave gratuita em "
                "api.openverse.org/v1/auth_tokens/register e ponha em OPENVERSE_TOKEN")
    if codigo == 429:
        return "pediram para ir mais devagar (429)"
    return str(exc)


MIN_LADO = 512          # imagem pequena nao ensina estilo, so borra

# "todas" inclui licenca que proibe uso comercial; a escolha e sua e fica registrada
LICENCAS = {
    "livres": "cc0,pdm,by,by-sa",
    "sem-comercial": "cc0,pdm,by,by-sa,by-nc,by-nc-sa",
    "todas": "",
}

_job: dict = {"state": "idle", "linhas": [], "baixadas": 0, "alvo": 0, "pasta": "", "termo": ""}


def _diz(msg: str) -> None:
    _job["linhas"].append(msg)
    del _job["linhas"][:-80]


def status() -> dict:
    return {**_job, "colecoes": colecoes()}


def raiz() -> Path:
    path = resolve("data/imagens")
    path.mkdir(parents=True, exist_ok=True)
    return path


def colecoes() -> list[dict]:
    saida = []
    for pasta in sorted(raiz().iterdir()) if raiz().exists() else []:
        if not pasta.is_dir():
            continue
        imgs = [p for p in pasta.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        meta = pasta / "creditos.json"
        dados = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        legendas = pasta / "legendas.txt"
        saida.append({
            "id": pasta.name,
            "imagens": len(imgs),
            "bytes": sum(p.stat().st_size for p in imgs),
            "termo": dados.get("termo", ""),
            "licenca": dados.get("licenca", ""),
            "tem_legendas": legendas.exists(),
        })
    return saida


def _busca_openverse(termo: str, quantidade: int, licenca: str) -> list[dict]:
    achados, pagina = [], 1
    while len(achados) < quantidade and pagina <= 5:
        params = {"q": termo, "page_size": 50, "page": pagina}
        if LICENCAS.get(licenca):
            params["license"] = LICENCAS[licenca]
        url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(params)
        try:
            dados = json_de(url)
        except Exception as exc:
            _diz(f"openverse: {_humano(exc)}")
            break
        for item in dados.get("results") or []:
            if min(item.get("width") or 0, item.get("height") or 0) < MIN_LADO:
                continue
            achados.append({
                "url": item.get("url"), "fonte": "openverse",
                "autor": item.get("creator") or "?", "licenca": item.get("license") or "?",
                "pagina": item.get("foreign_landing_url") or "", "titulo": item.get("title") or "",
            })
        if not dados.get("results"):
            break
        pagina += 1
    return achados[:quantidade]


def _busca_commons(termo: str, quantidade: int) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {termo}", "gsrnamespace": "6",
        "gsrlimit": str(min(quantidade * 2, 100)),
        "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": "1024",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        dados = json_de(url)
    except Exception as exc:
        _diz(f"commons: {_humano(exc)}")
        return []
    achados = []
    for pag in ((dados.get("query") or {}).get("pages") or {}).values():
        ii = (pag.get("imageinfo") or [{}])[0]
        if min(ii.get("width") or 0, ii.get("height") or 0) < MIN_LADO:
            continue
        extra = ii.get("extmetadata") or {}
        achados.append({
            "url": ii.get("thumburl") or ii.get("url"), "fonte": "wikimedia",
            "autor": re.sub(r"<[^>]+>", "", (extra.get("Artist") or {}).get("value", "?"))[:80],
            "licenca": (extra.get("LicenseShortName") or {}).get("value", "?"),
            "pagina": ii.get("descriptionurl") or "", "titulo": pag.get("title", ""),
        })
    return achados[:quantidade]


def coletar(termo: str, quantidade: int = 24, licenca: str = "livres",
            fontes: str = "openverse,wikimedia", nome: str = "") -> dict:
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma coleta rodando"}
    termo = (termo or "").strip()
    if not termo:
        return {"accepted": False, "reason": "diga o estilo que voce procura"}
    quantidade = max(4, min(int(quantidade or 24), 200))
    pasta = raiz() / (slug(nome) or slug(termo) or "colecao")
    pasta.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "baixadas": 0, "alvo": quantidade,
                 "pasta": str(pasta), "termo": termo})

    def _go() -> None:
        try:
            achados: list[dict] = []
            if "openverse" in fontes:
                achados += _busca_openverse(termo, quantidade, licenca)
                _diz(f"openverse: {len(achados)} candidata(s)")
            if "wikimedia" in fontes and len(achados) < quantidade:
                extras = _busca_commons(termo, quantidade - len(achados))
                achados += extras
                _diz(f"wikimedia: +{len(extras)} candidata(s)")
            if not achados:
                raise RuntimeError("nada encontrado com essa licenca; tente outro termo ou solte a licenca")
            creditos = []
            vistos = set()
            for item in achados:
                if _job["state"] != "running" or _job["baixadas"] >= quantidade:
                    break
                url = item["url"]
                if not url or url in vistos:
                    continue
                vistos.add(url)
                ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"
                alvo = pasta / f"{_job['baixadas'] + 1:03d}{ext}"
                try:
                    dados = abrir(url, timeout=60)
                    if len(dados) < 15_000:
                        continue
                    alvo.write_bytes(dados)
                except Exception as exc:
                    _diz(f"pulei {url[:50]}: {_humano(exc)}")
                    continue
                _job["baixadas"] += 1
                creditos.append({**item, "arquivo": alvo.name})
                _diz(f"{_job['baixadas']}/{quantidade} — {item['licenca']} — {item['autor'][:30]}")
            (pasta / "creditos.json").write_text(json.dumps({
                "termo": termo, "licenca": licenca, "fontes": fontes,
                "baixado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "aviso": "Cite autor e licenca de cada imagem se publicar algo treinado com elas.",
                "itens": creditos,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            _job["state"] = "done"
            _diz(f"COLETA CONCLUIDA — {_job['baixadas']} imagem(ns) em {pasta}")
        except Exception as exc:
            _job["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="coletor", daemon=True).start()
    return {"accepted": True, "pasta": str(pasta)}


def parar() -> dict:
    if _job["state"] == "running":
        _job["state"] = "done"
        _diz("parado por voce")
    return status()


def apagar(nome: str) -> dict:
    import shutil

    alvo = raiz() / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "colecao nao encontrada"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "colecoes": colecoes()}
