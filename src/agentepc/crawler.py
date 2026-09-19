"""Le um site de documentacao inteiro e devolve texto limpo para virar arquivo de treino.

Sites de doc dividem o conteudo em abas e links; aqui a raiz e so o ponto de partida.
O extrator segue os links do mesmo dominio, pagina por pagina, com limite e pausa —
e um visitante educado, nao um aspirador.

Paginas com conteudo montado por JavaScript nao aparecem no HTML cru; por isso existe o
modo navegador, que abre a pagina no Chrome sem janela e pega o DOM ja renderizado.
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from agentepc import ollama
from agentepc.config import load, resolve

UA = "Mozilla/5.0 (compatible; agente-pc/1.0; +local training data collector)"
SKIP_EXT = (".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
            ".mp4", ".mp3", ".css", ".js", ".json", ".xml", ".ico", ".woff", ".woff2")

# A extracao pode render dezenas de MB: nada disso fica em memoria nem sobe para a pagina.
# Cada pagina e gravada no arquivo assim que sai, e a tela mostra so contadores e amostra.
_job: dict = {
    "state": "idle", "url": "", "pages": 0, "target": 0, "images": 0, "queue": 0, "externas": 0,
    "chars": 0, "file": "", "preview": "", "lines": [], "visited": [],
}


def _say(msg: str) -> None:
    _job["lines"].append(msg)
    del _job["lines"][:-60]


def status() -> dict:
    return {k: v for k, v in _job.items() if k != "visited"}


def out_dir() -> Path:
    path = resolve("data/extracoes")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ navegador
def _chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    for path in (
        "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
    ):
        if Path(path).exists():
            return path
    return None


def _fetch_browser(url: str, binary: str, timeout: int = 60) -> str:
    proc = subprocess.run(
        [binary, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=5000", "--dump-dom", url],
        capture_output=True, text=True, errors="replace", timeout=timeout,
    )
    return proc.stdout or ""


def _fetch_plain(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype:
            return ""
        return resp.read(3_000_000).decode(resp.headers.get_content_charset() or "utf-8", "replace")


# -------------------------------------------------------------------- extracao
def _strip(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw)


def _clean_text(chunk: str) -> str:
    # <code> vira `crase`: assim o comando chega marcado ao formatador e nao vira texto solto
    chunk = re.sub(r"(?is)<code\b[^>]*>(.*?)</code>", lambda m: "`" + _strip(m.group(1)).strip() + "`", chunk)
    text = html.unescape(_strip(chunk))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return text.strip()


def extract(page: str) -> tuple[str, str, list[dict]]:
    """Devolve (titulo, texto, imagens). Mantem cabecalhos, listas e blocos de codigo."""
    body = re.sub(r"(?is)<(script|style|noscript|svg|template)[^>]*>.*?</\1>", " ", page)
    title = ""
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    if m:
        title = _clean_text(m.group(1))

    images = []
    for tag in re.findall(r"(?is)<img\b[^>]*>", body):
        src = re.search(r'(?is)\bsrc\s*=\s*["\']([^"\']+)', tag)
        alt = re.search(r'(?is)\balt\s*=\s*["\']([^"\']*)', tag)
        if src and not src.group(1).startswith("data:"):
            images.append({"src": src.group(1), "alt": _clean_text(alt.group(1) if alt else "")})

    # o miolo da doc costuma estar em <main> ou <article>; sem isso, sobra o <body>
    m = re.search(r"(?is)<(main|article)\b[^>]*>(.*?)</\1>", body)
    core = m.group(2) if m else (re.search(r"(?is)<body\b[^>]*>(.*?)</body>", body) or re.match(r"(.*)", body, re.S)).group(1)
    core = re.sub(r"(?is)<(nav|footer|aside|form)\b[^>]*>.*?</\1>", " ", core)

    out: list[str] = []
    for kind, raw in re.findall(r"(?is)<(h[1-6]|p|li|pre|td|th|figcaption|dt|dd)\b[^>]*>(.*?)</\1>", core):
        if kind == "pre":
            code = html.unescape(_strip(raw)).strip("\n")
            if code.strip():
                out.append("```\n" + code + "\n```")
            continue
        text = _clean_text(raw)
        if not text:
            continue
        if kind.startswith("h"):
            out.append("\n" + "#" * int(kind[1]) + " " + text)
        elif kind in ("li", "dt", "dd"):
            out.append("- " + text)
        else:
            out.append(text)
    seen, lines = set(), []
    for line in out:
        if line not in seen or line.startswith("```"):
            seen.add(line)
            lines.append(line)
    corpo = "\n".join(lines).strip()
    cmds = commands(corpo)
    if cmds:
        # lista explicita: garante que nenhum comando da pagina se perca no meio do texto
        corpo += "\n\n### Comandos desta pagina\n" + "\n".join(f"- `{c}`" for c in cmds)
    return title, corpo, images


CMD_RE = re.compile(r"`([a-z][\w.-]*(?:\s+[\w.:=@/+-]+)+)`")


def commands(text: str) -> list[str]:
    """Comandos citados no texto: entre crases ou dentro de bloco de codigo."""
    achados: list[str] = []
    for linha in re.findall(r"```(.*?)```", text, re.S):
        for l in linha.splitlines():
            l = l.strip().lstrip("$ ").strip()
            if re.match(r"^[a-z][\w.-]*(\s+[\w.:=@/+-]+)+$", l) and len(l) < 120:
                achados.append(l)
    achados += [m.strip() for m in CMD_RE.findall(text) if len(m) < 120]
    vistos, saida = set(), []
    for c in achados:
        if c.lower() not in vistos:
            vistos.add(c.lower())
            saida.append(c)
    return saida[:200]


def _links(page: str, base: str, root: str, externos: bool = False) -> list[tuple[str, bool]]:
    """(url, e_de_fora). Doc de instalacao aponta para a doc de outra ferramenta; com
    externos ligados esses links entram, mas so um nivel — nao viram um rastejador da web."""
    host = urllib.parse.urlparse(root).netloc
    found = []
    for href in re.findall(r'(?is)<a\b[^>]*\bhref\s*=\s*["\']([^"\']+)', page):
        url = urllib.parse.urljoin(base, html.unescape(href)).split("#")[0].rstrip("/")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or url.lower().endswith(SKIP_EXT):
            continue
        fora = parsed.netloc != host
        if fora and not externos:
            continue
        found.append((url, fora))
    return found


# --------------------------------------------------------------------- imagens
def vision_model() -> str:
    name = (load().get("learn") or {}).get("vision_model") or ""
    if not name:
        return ""
    try:
        tags = ollama.tags()
    except Exception:
        return ""
    return name if any(t == name or t.startswith(name.split(":")[0] + ":") for t in tags) else ""


_VAZIO = ("nao contem informacoes", "não contém informações", "powered by", "nao ha informacoes",
          "não há informações", "imagem em branco", "nao e possivel", "não é possível")


def _util(desc: str) -> bool:
    """Logo, banner e icone rendem descricao vazia; isso viraria ruido no treino."""
    low = (desc or "").strip().lower()
    return len(low) > 25 and not any(v in low for v in _VAZIO)


def describe_image(url: str, alt: str, model: str) -> str:
    """Manda a imagem para o modelo de visao. Sem modelo, sobra o texto alternativo."""
    if not model:
        return alt
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if int(resp.headers.get("Content-Length") or 0) > 6_000_000:
                return alt
            raw = resp.read(6_000_000)
        data = ollama.request(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0, "num_predict": 120},
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Descreva em portugues, em ate 2 frases, o que esta imagem de documentacao ensina. "
                            "Se for diagrama ou print de tela, diga os elementos e o fluxo. Sem rodeios."
                            + (f" Legenda existente: {alt}" if alt else "")
                        ),
                        "images": [base64.b64encode(raw).decode()],
                    }
                ],
            },
            timeout=300,
        )
        return ((data.get("message") or {}).get("content") or "").strip() or alt
    except Exception:
        return alt


# ----------------------------------------------------------------------- job
def start(url: str, max_pages: int = 0, browser: bool = True, images: bool = True,
          externos: bool = False) -> dict:
    """max_pages 0 = sem limite: anda a arvore inteira do dominio ate a fila esvaziar."""
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma extracao rodando"}
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        return {"accepted": False, "reason": "informe a URL inteira, com http:// ou https://"}
    max_pages = max(0, int(max_pages or 0))
    binary = _chrome() if browser else None
    if browser and not binary:
        return {"accepted": False, "reason": "Chrome nao encontrado; desmarque 'usar navegador'"}
    vmodel = vision_model() if images else ""
    host = urllib.parse.urlparse(url).netloc.replace(":", "-")
    dest = out_dir() / f"{host}-{time.strftime('%Y%m%d-%H%M%S')}.md"
    dest.write_text(f"# Extracao de {url}\n", encoding="utf-8")
    _job.update({
        "state": "running", "url": url, "pages": 0, "target": max_pages, "images": 0, "queue": 1, "externas": 0,
        "chars": 0, "file": str(dest), "preview": "", "lines": [], "visited": [],
    })

    def _go() -> None:
        try:
            if images and not vmodel:
                _say("sem modelo de visao instalado: uso so a legenda das imagens (veja learn.vision_model)")
            elif vmodel:
                _say(f"imagens vao passar pelo {vmodel}")
            _say(("navegador (Chrome sem janela): pega paginas montadas por JavaScript"
                  if binary else "modo simples: le o HTML cru, sem JavaScript"))
            _say("sem limite de paginas" if not max_pages else f"limite de {max_pages} paginas")
            _say("seguindo tambem links para fora do dominio (1 nivel)" if externos
                 else "so links do mesmo dominio")
            root = url.rstrip("/")
            queue, seen, conteudos = [(root, False)], {root}, set()
            delay = float((load().get("learn") or {}).get("crawl_delay") or 0.5)
            t0 = time.time()
            while queue and _job["state"] == "running":
                if max_pages and _job["pages"] >= max_pages:
                    break
                page_url, de_fora = queue.pop(0)
                _job["queue"] = len(queue)
                try:
                    page = _fetch_browser(page_url, binary) if binary else _fetch_plain(page_url)
                except (urllib.error.URLError, OSError, subprocess.SubprocessError, ValueError) as exc:
                    _say(f"pulei {page_url}: {exc}")
                    continue
                if not page.strip():
                    continue
                title, text, imgs = extract(page)
                _job["pages"] += 1
                _job["externas"] += 1 if de_fora else 0
                _job["visited"].append(page_url)
                # a mesma pagina costuma ter varias URLs; o conteudo decide se ja veio
                marca = hash(text[:4000])
                if text and marca not in conteudos:
                    conteudos.add(marca)
                    block = [f"\n\n## {title or page_url}", f"fonte: {page_url}", text]
                    for img in imgs[:8] if vmodel else []:
                        full = urllib.parse.urljoin(page_url, img["src"])
                        desc = describe_image(full, img["alt"], vmodel)
                        if _util(desc):
                            _job["images"] += 1
                            block.append(f"[imagem] {desc}")
                    with Path(_job["file"]).open("a", encoding="utf-8") as fh:
                        fh.write("\n".join(block) + "\n")
                    _job["chars"] = Path(_job["file"]).stat().st_size
                    if len(_job["preview"]) < 2000:
                        _job["preview"] += "\n".join(block)[:2000]
                ritmo = _job["pages"] / max(time.time() - t0, 1)
                _say(f"{_job['pages']} lidas, {len(queue)} na fila ({ritmo * 60:.0f}/min) — {(title or page_url)[:60]}")
                # pagina de fora nao espalha: entra, e o galho para ali
                for link, fora in ([] if de_fora else _links(page, page_url, root, externos)):
                    if link not in seen:
                        seen.add(link)
                        queue.append((link, fora))
                time.sleep(delay)
            _job["queue"] = len(queue)
            _job["state"] = "done"
            _say(f"fim: {_job['pages']} paginas, {_job['chars'] / 1000:.0f} mil caracteres, "
                 f"{_job['images']} imagens -> {_job['file']}")
        except Exception as exc:
            _job["state"] = "error"
            _say(f"erro: {exc}")

    threading.Thread(target=_go, name="crawler", daemon=True).start()
    return {"accepted": True, "browser": bool(binary), "vision": vmodel, "file": str(dest)}


def stop() -> dict:
    if _job["state"] == "running":
        _job["state"] = "done"
        _say("parado por voce; o que ja foi lido esta salvo no arquivo")
    return status()
