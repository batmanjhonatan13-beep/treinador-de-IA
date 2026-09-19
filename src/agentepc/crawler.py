"""Le um site de documentacao e devolve texto limpo para virar arquivo de treino.

Em dois tempos:

1. **descobrir** — anda a arvore a partir da raiz e lista cada pagina com o titulo,
   guardando o texto de lado. Filtros de endereco mantem o passeio dentro do que
   interessa: so a versao 3 do Python, so o espaco de um time no Confluence.
2. **montar** — voce marca as paginas que quer e so elas viram o arquivo de treino.

Assim nao e preciso adivinhar um numero de paginas: voce ve os nomes e escolhe.

Pagina montada por JavaScript nao aparece no HTML cru; por isso existe o modo navegador,
que abre a pagina no Chrome sem janela e pega o DOM ja renderizado.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from agentepc import lotes, ollama
from agentepc.config import load, resolve

UA = "Mozilla/5.0 (compatible; agente-pc/1.0; +local training data collector)"
SKIP_EXT = (".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
            ".mp4", ".mp3", ".css", ".js", ".json", ".xml", ".ico", ".woff", ".woff2")

# O texto pode dar dezenas de MB: fica em disco (paginas.jsonl), nunca na resposta da API.
# Para a tela vai so a lista de paginas (endereco, titulo, tamanho).
_job: dict = {
    "state": "idle", "url": "", "pages": 0, "target": 0, "images": 0, "queue": 0, "externas": 0,
    "chars": 0, "file": "", "cache": "", "lines": [], "paginas": [], "visited": [],
    "fase": "", "selecionadas": 0, "lote": "", "terminou": False,
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


# A tela de "prove que voce nao e um robo" volta com HTTP 200 e um corpo bonito: sem
# reconhece-la, o extrator soma 2.000 paginas lidas e entrega meia duzia — foi o que
# aconteceu com a doc da Godot aqui. Estes sao os textos que essas telas usam.
VERIFICACAO = re.compile(
    r"(?i)just a moment|um momento|verificaç[aã]o de seguran|checking your browser|"
    r"enable javascript and cookies|cf-browser-verification|cf_chl_opt|challenge-platform|"
    r"attention required!|acesso negado pelo firewall"
)


def parece_verificacao(page: str, titulo: str) -> bool:
    """A pagina veio, mas e a tela de verificacao do site, nao o conteudo."""
    if VERIFICACAO.search(titulo or ""):
        return True
    # o corpo de uma tela dessas e pequeno; uma pagina de doc de verdade e bem maior
    return len(page) < 80_000 and bool(VERIFICACAO.search(page))


def _fetch_browser(url: str, binary: str, timeout: int = 60) -> str:
    proc = subprocess.run(
        [binary, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=5000", "--dump-dom", url],
        capture_output=True, text=True, errors="replace", timeout=timeout,
    )
    return proc.stdout or ""


AVISO_BARRADO = (
    "PAREI: o site {motivo} em vez de entregar a pagina.\n"
    "Nao e defeito do extrator nem culpa sua: muitos sites cortam quem le muitas paginas "
    "seguidas.\n"
    "O que fazer:\n"
    "  1. espere uns minutos e aumente o intervalo entre paginas (crawl_delay no config.yaml);\n"
    "  2. melhor ainda: quase toda documentacao tem o pacote pronto para baixar. No Read the "
    "Docs e .../_/downloads/<idioma>/<versao>/htmlzip/ — use 'Importar documentacao (.zip)' "
    "aqui na pagina: baixa uma vez so e nao incomoda o site;\n"
    "  3. as paginas lidas ate agora ficaram salvas e ja dao para marcar e extrair."
)


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


def normaliza(url: str) -> str:
    """Mesma pagina por caminhos diferentes vira um endereco so.

    Tira a ancora, o index.html e a barra final: /tutorial/, /tutorial e
    /tutorial/index.html sao a mesma leitura.
    """
    url = url.split("#")[0].strip()
    partes = urllib.parse.urlsplit(url)
    caminho = re.sub(r"/index\.html?$", "/", partes.path)
    if caminho.endswith("/") and len(caminho) > 1:
        caminho = caminho[:-1]
    return urllib.parse.urlunsplit((partes.scheme, partes.netloc, caminho or "/", partes.query, ""))


def _termos(texto: str) -> list[str]:
    """Aceita termos separados por virgula, ponto-e-virgula ou quebra de linha."""
    return [x.strip() for x in re.split(r"[,;\n]", texto or "") if x.strip()]


def passa_filtro(url: str, incluir: list[str], excluir: list[str], prefixo: str = "") -> bool:
    """Endereco decide o assunto: /pt-br/3/ e uma versao do Python, /spaces/TIME/ e um time."""
    baixo = url.lower()
    if prefixo:
        # a barra e a fronteira: sem ela, /pt-br/3 deixaria passar /pt-br/3.13
        base = prefixo.rstrip("/").lower()
        if baixo != base and not baixo.startswith(base + "/"):
            return False
    if incluir and not any(t.lower() in baixo for t in incluir):
        return False
    return not any(t.lower() in baixo for t in excluir)


def _links(page: str, base: str, root: str, externos: bool = False) -> list[tuple[str, bool]]:
    """(url, e_de_fora). Doc de instalacao aponta para a doc de outra ferramenta; com
    externos ligados esses links entram, mas so um nivel — nao viram um rastejador da web."""
    host = urllib.parse.urlparse(root).netloc
    found = []
    for href in re.findall(r'(?is)<a\b[^>]*\bhref\s*=\s*["\']([^"\']+)', page):
        url = normaliza(urllib.parse.urljoin(base, html.unescape(href)))
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
def _cache_path(base: Path) -> Path:
    return base.with_suffix(".paginas.jsonl")


def descobrir(url: str, browser: bool = True, externos: bool = False, max_pages: int = 0,
              incluir: str = "", excluir: str = "", so_abaixo: bool = True, assunto: str = "") -> dict:
    """Anda a arvore e lista as paginas. Nao monta arquivo de treino ainda."""
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma busca rodando"}
    from agentepc import formatter

    if formatter.status()["state"] in ("running", "comandos", "parando"):
        return {"accepted": False, "reason": "tem um lote sendo formatado; espere ou pare a formatacao"}
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        return {"accepted": False, "reason": "informe a URL inteira, com http:// ou https://"}
    binary = _chrome() if browser else None
    if browser and not binary:
        return {"accepted": False, "reason": "Chrome nao encontrado; desmarque 'usar navegador'"}
    max_pages = max(0, int(max_pages or 0))
    termos_sim, termos_nao = _termos(incluir), _termos(excluir)
    # a barra final importa: sem ela, "tutorial/x.html" resolveria um nivel acima
    raiz = url if url.endswith("/") or "." in url.rsplit("/", 1)[-1] else url + "/"
    # "so abaixo deste caminho" resolve o caso das versoes: a raiz
    # https://docs.python.org/pt-br/3/ nao deixa entrar /pt-br/3.13/
    prefixo = raiz.rstrip("/") if so_abaixo else ""
    lote, destino = lotes.caminho_livre(lotes.nome_para(assunto, url))
    cache = _cache_path(destino)
    cache.write_text("", encoding="utf-8")
    lotes.gravar_meta(lote, {"nome": lote, "assunto": assunto, "url": url, "paginas": 0, "chars": 0})
    _job.update({
        "state": "running", "fase": "descobrindo", "url": url, "pages": 0, "target": max_pages,
        "images": 0, "queue": 1, "externas": 0, "chars": 0, "file": str(destino),
        "cache": str(cache), "lines": [], "paginas": [], "visited": [], "selecionadas": 0,
        "lote": lote, "terminou": False,
    })

    def _go() -> None:
        try:
            _say("navegador (Chrome sem janela): pega paginas montadas por JavaScript"
                 if binary else "modo simples: le o HTML cru, sem JavaScript")
            if prefixo:
                _say(f"so paginas abaixo de {prefixo}")
            if termos_sim:
                _say(f"so endereco contendo: {', '.join(termos_sim)}")
            if termos_nao:
                _say(f"pulando endereco com: {', '.join(termos_nao)}")
            fila, vistos, conteudos = [(raiz, False)], {normaliza(raiz)}, set()
            barrados = 0
            espera = float((load().get("learn") or {}).get("crawl_delay") or 0.5)
            t0 = time.time()
            with Path(_job["cache"]).open("a", encoding="utf-8") as fh:
                while fila and _job["state"] == "running":
                    if max_pages and _job["pages"] >= max_pages:
                        break
                    pagina_url, de_fora = fila.pop(0)
                    _job["queue"] = len(fila)
                    try:
                        page = _fetch_browser(pagina_url, binary) if binary else _fetch_plain(pagina_url)
                    except (urllib.error.URLError, OSError, subprocess.SubprocessError, ValueError) as exc:
                        if getattr(exc, "code", None) == 429:
                            barrados += 1
                            if barrados >= 3:
                                _say(AVISO_BARRADO.format(motivo="pediu para ir mais devagar (429)"))
                                break
                            _say("o site pediu calma (429); esperando 30 s")
                            time.sleep(30)
                            fila.insert(0, (pagina_url, de_fora))
                            continue
                        _say(f"pulei {pagina_url}: {exc}")
                        continue
                    if not page.strip():
                        continue
                    titulo, texto, imgs = extract(page)
                    if parece_verificacao(page, titulo):
                        # nao conta como lida: contar mentiria no numero e esconderia o problema
                        barrados += 1
                        if barrados >= 3:
                            _say(AVISO_BARRADO.format(
                                motivo=f"devolveu a tela de verificacao (\"{titulo[:30]}\")"))
                            break
                        _say(f"verificacao de robo em {pagina_url}; tentando a proxima")
                        time.sleep(5)
                        continue
                    barrados = 0
                    _job["pages"] += 1
                    _job["externas"] += 1 if de_fora else 0
                    _job["visited"].append(pagina_url)
                    marca = hash(texto[:4000])
                    if texto and marca not in conteudos:
                        conteudos.add(marca)
                        fh.write(json.dumps({"url": pagina_url, "titulo": titulo, "texto": texto,
                                             "imagens": imgs[:8], "fora": de_fora},
                                            ensure_ascii=False) + "\n")
                        fh.flush()
                        _job["chars"] += len(texto)
                        _job["paginas"].append({
                            "url": pagina_url, "titulo": titulo or pagina_url,
                            "chars": len(texto), "imagens": len(imgs), "fora": de_fora,
                        })
                    ritmo = _job["pages"] / max(time.time() - t0, 1)
                    _say(f"{_job['pages']} lidas, {len(fila)} na fila ({ritmo * 60:.0f}/min) — "
                         f"{(titulo or pagina_url)[:60]}")
                    for link, fora in ([] if de_fora else _links(page, pagina_url, raiz, externos)):
                        if link in vistos:
                            continue
                        if not fora and not passa_filtro(link, termos_sim, termos_nao, prefixo):
                            continue
                        vistos.add(link)
                        fila.append((link, fora))
                    time.sleep(espera)
            _job["queue"] = len(fila)
            _job["state"] = "done"
            _job["fase"] = "descoberto"
            _job["terminou"] = True
            lotes.gravar_meta(_job["lote"], {"paginas": len(_job["paginas"]), "chars": _job["chars"]})
            _say(f"BUSCA CONCLUIDA — {len(_job['paginas'])} pagina(s) encontradas"
                 + (f", {len(fila)} link(s) ficaram de fora pelo teto" if fila else "")
                 + ". Marque as que quer e clique em Extrair.")
        except Exception as exc:
            _job["state"] = "error"
            _say(f"erro: {exc}")

    threading.Thread(target=_go, name="crawler", daemon=True).start()
    return {"accepted": True, "browser": bool(binary), "file": str(destino)}



IGNORAR_NOMES = {"search.html", "genindex.html", "py-modindex.html", "404.html"}
IGNORAR_PASTAS = ("/_static/", "/_sources/", "/_images/", "/node_modules/")


def _serve_como_pagina(nome: str) -> bool:
    baixo = nome.lower()
    if not baixo.endswith((".html", ".htm")):
        return False
    if Path(baixo).name in IGNORAR_NOMES:
        return False
    return not any(parte in "/" + baixo for parte in IGNORAR_PASTAS)


def _percorre_pacote(caminho: Path):
    """Devolve (nome, html) de um .zip, de um .tar.* ou de uma pasta.

    Sao tres formatos porque e assim que a documentacao aparece no mundo real: o Python e o
    Django publicam .zip, os conjuntos do Dash (que tem Docker, Ansible, PostgreSQL) vem em
    .tgz, e as vezes voce ja tem a doc descompactada numa pasta.
    """
    import tarfile
    import zipfile

    if caminho.is_dir():
        for arq in sorted(caminho.rglob("*")):
            if arq.is_file() and _serve_como_pagina(str(arq)):
                yield str(arq.relative_to(caminho)), arq.read_bytes()
        return
    if zipfile.is_zipfile(caminho):
        with zipfile.ZipFile(caminho) as zf:
            for nome in sorted(zf.namelist()):
                if _serve_como_pagina(nome):
                    yield nome, zf.read(nome)
        return
    if tarfile.is_tarfile(caminho):
        # leitura em fluxo: num .tgz de 200 MB, voltar atras para cada arquivo seria lento
        with tarfile.open(caminho, "r|*") as tf:
            for membro in tf:
                if membro.isfile() and _serve_como_pagina(membro.name):
                    peca = tf.extractfile(membro)
                    if peca:
                        yield membro.name, peca.read()
        return
    raise RuntimeError("nao reconheci o arquivo: esperava .zip, .tar.gz/.tgz ou uma pasta")


def importar_zip(origem: str, assunto: str = "") -> dict:
    """Importa documentacao ja empacotada, em vez de andar pelo site.

    E o caminho certo para doc grande: um download em vez de milhares de requisicoes, nada
    de tela de verificacao de robo, e o conteudo vem completo. Onde achar:

      * Python, Django e outros publicam .zip da doc no proprio site;
      * projetos no Read the Docs: .../_/downloads/<idioma>/<versao>/htmlzip/ ;
      * os conjuntos do Dash (kapeli.com/feeds/<Nome>.tgz) cobrem o que nao publica nada,
        como Docker, Ansible e PostgreSQL;
      * ou aponte uma pasta que voce ja descompactou aqui na maquina.

    Depois de importar, a lista de paginas aparece igual a de uma busca: marque e extraia.
    """
    import tempfile

    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma busca rodando"}
    from agentepc import formatter

    if formatter.status()["state"] in ("running", "comandos", "parando"):
        return {"accepted": False, "reason": "tem um lote sendo formatado; espere ou pare a formatacao"}
    origem = (origem or "").strip()
    if not origem:
        return {"accepted": False, "reason": "informe o endereco do pacote ou o caminho aqui na maquina"}
    lote, destino = lotes.caminho_livre(lotes.nome_para(assunto, origem))
    cache = _cache_path(destino)
    cache.write_text("", encoding="utf-8")
    lotes.gravar_meta(lote, {"nome": lote, "assunto": assunto, "url": origem, "paginas": 0, "chars": 0})
    _job.update({
        "state": "running", "fase": "importando", "url": origem, "pages": 0, "target": 0,
        "images": 0, "queue": 0, "externas": 0, "chars": 0, "file": str(destino),
        "cache": str(cache), "lines": [], "paginas": [], "visited": [], "selecionadas": 0,
        "lote": lote, "terminou": False,
    })

    def _go() -> None:
        temporario = None
        try:
            if re.match(r"^https?://", origem):
                _say(f"baixando {origem}")
                pedido = urllib.request.Request(origem, headers={"User-Agent": UA})
                with urllib.request.urlopen(pedido, timeout=300) as resp:
                    total = int(resp.headers.get("Content-Length") or 0)
                    temporario = Path(tempfile.mkstemp(suffix=".pacote")[1])
                    baixado = 0
                    with temporario.open("wb") as saida:
                        while pedaco := resp.read(1 << 20):
                            saida.write(pedaco)
                            baixado += len(pedaco)
                            if total and baixado % (20 << 20) < (1 << 20):
                                _say(f"baixando: {baixado / 1e6:.0f} de {total / 1e6:.0f} MB")
                caminho = temporario
                _say(f"baixado: {caminho.stat().st_size / 1e6:.1f} MB")
            else:
                caminho = Path(origem).expanduser()
                if not caminho.exists():
                    raise RuntimeError(f"nao achei {caminho}")
            conteudos = set()
            achou = False
            with Path(_job["cache"]).open("a", encoding="utf-8") as fh:
                for nome, bruto in _percorre_pacote(caminho):
                    achou = True
                    if _job["state"] != "running":
                        break
                    titulo, texto, imgs = extract(bruto.decode("utf-8", "replace"))
                    _job["pages"] += 1
                    marca = hash(texto[:4000])
                    if not texto or marca in conteudos:
                        continue
                    conteudos.add(marca)
                    endereco = (f"{origem.rstrip('/')}#{nome}" if re.match(r"^https?://", origem)
                                else str(Path(origem) / nome))
                    fh.write(json.dumps({"url": endereco, "titulo": titulo, "texto": texto,
                                         "imagens": imgs[:8], "fora": False},
                                        ensure_ascii=False) + "\n")
                    _job["chars"] += len(texto)
                    _job["paginas"].append({
                        "url": endereco, "titulo": titulo or nome,
                        "chars": len(texto), "imagens": len(imgs), "fora": False,
                    })
                    if _job["pages"] % 50 == 0:
                        _say(f"{_job['pages']} lidas — {(titulo or nome)[:60]}")
            if not achou:
                raise RuntimeError("nao achei pagina HTML dentro desse pacote")
            _job["state"] = "done"
            _job["fase"] = "descoberto"
            _job["terminou"] = True
            lotes.gravar_meta(_job["lote"], {"paginas": len(_job["paginas"]), "chars": _job["chars"]})
            _say(f"IMPORTACAO CONCLUIDA — {len(_job['paginas'])} pagina(s) com texto de "
                 f"{_job['pages']} arquivo(s). Marque as que quer e clique em Extrair.")
        except Exception as exc:
            _job["state"] = "error"
            _say(f"erro: {exc}")
        finally:
            if temporario and temporario.exists():
                temporario.unlink()

    threading.Thread(target=_go, name="importar-pacote", daemon=True).start()
    return {"accepted": True, "lote": lote}


def montar(urls: list[str], imagens: bool = True) -> dict:
    """Escreve o arquivo de treino so com as paginas marcadas."""
    if _job["state"] == "running":
        return {"accepted": False, "reason": "espere a busca terminar"}
    cache = Path(_job.get("cache") or "")
    if not cache.exists():
        return {"accepted": False, "reason": "nao ha busca para aproveitar; descubra as paginas antes"}
    escolhidas = set(urls or [])
    if not escolhidas:
        return {"accepted": False, "reason": "marque ao menos uma pagina"}
    vmodel = vision_model() if imagens else ""
    _job.update({"state": "running", "fase": "montando", "images": 0,
                 "selecionadas": len(escolhidas), "lines": [], "terminou": False})

    def _go() -> None:
        try:
            destino = Path(_job["file"])
            _say(f"montando {len(escolhidas)} pagina(s)"
                 + (f"; imagens pelo {vmodel}" if vmodel else ""))
            with destino.open("w", encoding="utf-8") as saida:
                saida.write(f"# Extracao de {_job['url']}\n")
                feitas = 0
                for linha in cache.read_text(encoding="utf-8").splitlines():
                    if not linha.strip() or _job["state"] != "running":
                        continue
                    reg = json.loads(linha)
                    if reg["url"] not in escolhidas:
                        continue
                    bloco = [f"\n\n## {reg['titulo'] or reg['url']}", f"fonte: {reg['url']}", reg["texto"]]
                    for img in reg.get("imagens") or []:
                        if not vmodel:
                            break
                        full = urllib.parse.urljoin(reg["url"], img["src"])
                        desc = describe_image(full, img.get("alt", ""), vmodel)
                        if _util(desc):
                            _job["images"] += 1
                            bloco.append(f"[imagem] {desc}")
                    saida.write("\n".join(bloco) + "\n")
                    feitas += 1
                    _say(f"{feitas}/{len(escolhidas)} — {(reg['titulo'] or reg['url'])[:60]}")
            _job["chars"] = destino.stat().st_size
            _job["state"] = "done"
            _job["fase"] = "montado"
            _job["terminou"] = True
            lotes.gravar_meta(_job["lote"], {"paginas": feitas, "chars": _job["chars"],
                                             "imagens": _job["images"]})
            _say(f"pronto: {feitas} pagina(s), {_job['chars'] / 1000:.0f} mil caracteres, "
                 f"{_job['images']} imagem(ns) -> {destino}")
        except Exception as exc:
            _job["state"] = "error"
            _say(f"erro: {exc}")

    threading.Thread(target=_go, name="crawler-montar", daemon=True).start()
    return {"accepted": True}


def limpar() -> dict:
    """Esquece a busca atual (lista de paginas e cache), sem apagar lotes ja montados."""
    if _job["state"] == "running":
        return {"ok": False, "reason": "pare a busca antes de limpar"}
    # Path("") vira Path("."), que existe: sem esta guarda o "Limpar busca" sem busca
    # nenhuma tentava apagar a pasta atual
    cache = Path(_job["cache"]) if _job.get("cache") else None
    if cache and cache.is_file() and _job.get("fase") != "montado":
        cache.unlink()
        bruto = Path(_job["file"]) if _job.get("file") else None
        if bruto and bruto.is_file() and bruto.stat().st_size < 200:
            bruto.unlink()
            (bruto.parent / f"{bruto.stem}.json").unlink(missing_ok=True)
    _job.update({"state": "idle", "fase": "", "pages": 0, "queue": 0, "chars": 0, "images": 0,
                 "externas": 0, "paginas": [], "visited": [], "lines": [], "cache": "",
                 "file": "", "lote": "", "terminou": False, "selecionadas": 0})
    return {"ok": True}


def stop() -> dict:
    if _job["state"] == "running":
        _job["state"] = "done"
        _say("parado por voce; o que ja foi lido esta na lista")
    return status()
