from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agentepc import (capacidades, catalogo_docs, classificador, coletor, consolidado, crawler, export,
                      formatter, health, imagem, lotes, memory, ollama, provision, sistemas,
                      som, train, tresd)
from agentepc.config import ROOT

WEB = ROOT / "web"


def _fotos(limite: int = 200) -> list[str]:
    """Imagens que ja estao na maquina e servem de entrada para o 3D."""
    raiz = ROOT / "data"
    saida = []
    for pasta in ("geradas", "imagens", "amostras"):
        base = raiz / pasta
        if not base.is_dir():
            continue
        for arq in sorted(base.rglob("*")):
            if arq.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                saida.append(str(arq.relative_to(raiz)))
            if len(saida) >= limite:
                return saida
    return saida


def _audios(limite: int = 200) -> list[str]:
    """Audios que ja estao na maquina, para perguntar ao classificador."""
    raiz = ROOT / "data"
    saida = []
    for pasta in ("sons-gerados", "sons", "som-classes"):
        base = raiz / pasta
        if not base.is_dir():
            continue
        for arq in sorted(base.rglob("*")):
            if arq.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".oga", ".opus"):
                saida.append(str(arq.relative_to(raiz)))
            if len(saida) >= limite:
                return saida
    return saida


class Handler(BaseHTTPRequestHandler):
    server_version = "agente-pc"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            try:
                self._json(200, ollama.status())
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        query = parse_qs(urlparse(self.path).query)
        if path == "/api/train-log":
            try:
                self._json(200, train.read_log((query.get("id") or [""])[0]))
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            return
        if path == "/api/model-switch-job":
            self._json(200, health.troca_status())
            return
        if path == "/api/sistemas":
            self._json(200, {"itens": sistemas.nomes()})
            return
        if path == "/api/health":
            try:
                self._json(200, health.checar())
            except Exception as exc:
                self._json(200, {"conectado": False, "resumo": f"erro ao checar: {exc}", "itens": [], "modelos": []})
            return
        if path == "/api/prepare-job":
            self._json(200, provision.status())
            return
        if path == "/api/fix-job":
            self._json(200, formatter.conserto_status())
            return
        if path == "/api/consolidados":
            self._json(200, {"itens": consolidado.listar(), "regras": consolidado.REGRAS})
            return
        if path == "/api/gerar-job":
            self._json(200, imagem.geracao_status())
            return
        if path == "/api/instalar-job":
            self._json(200, capacidades.instalacao_status())
            return
        if path == "/api/capacidades":
            self._json(200, capacidades.avaliar())
            return
        if path == "/api/classificador":
            self._json(200, classificador.status())
            return
        if path == "/api/som":
            self._json(200, som.status())
            return
        if path == "/api/som-estilos":
            self._json(200, {"itens": som.estilos()})
            return
        if path == "/api/tresd":
            self._json(200, tresd.status())
            return
        if path == "/api/tresd-buscar":
            self._json(200, tresd.buscar((query.get("termo") or [""])[0]))
            return
        if path == "/api/fotos":
            self._json(200, {"itens": _fotos()})
            return
        if path == "/api/audios":
            self._json(200, {"itens": _audios()})
            return
        if path == "/api/imagens":
            self._json(200, {"coleta": coletor.status(), "treino": imagem.status()})
            return
        if path == "/api/doc-atualizar-job":
            self._json(200, catalogo_docs.status_atualizacao())
            return
        if path == "/api/doc-catalogo":
            self._json(200, catalogo_docs.resumo())
            return
        if path == "/api/lotes":
            self._json(200, {"itens": lotes.listar()})
            return
        if path == "/api/crawl-job":
            self._json(200, crawler.status())
            return
        if path == "/api/export-job":
            self._json(200, export.status())
            return
        if path == "/api/adapters":
            self._json(200, train.adapters())
            return
        if path == "/api/format-job":
            self._json(200, formatter.status())
            return
        if path == "/api/train-job":
            self._json(200, train.job_status())
            return
        if path == "/api/me":
            self._json(
                200,
                {
                    "profile": memory.read_profile(),
                    "files": memory.knowledge_index(),
                    "knowledge": memory.read_knowledge(),
                    "lora": train.adapter_exists(),
                },
            )
            return
        if path.startswith("/baixar/"):
            alvo = (ROOT / "data" / path[len("/baixar/"):]).resolve()
            permitido = (ROOT / "data").resolve()
            tipos = {
                ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
                ".flac": "audio/flac", ".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
                ".obj": "text/plain", ".ply": "application/octet-stream",
                ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
            }
            if permitido in alvo.parents and alvo.is_file() and alvo.suffix.lower() in tipos:
                dados = alvo.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", tipos[alvo.suffix.lower()])
                self.send_header("Content-Length", str(len(dados)))
                if alvo.suffix.lower() in (".glb", ".obj", ".ply", ".gltf"):
                    self.send_header("Content-Disposition", f'attachment; filename="{alvo.name}"')
                self.end_headers()
                self.wfile.write(dados)
            else:
                self.send_error(404)
            return
        if path.startswith("/img/"):
            alvo = (ROOT / "data" / path[len("/img/"):]).resolve()
            permitido = (ROOT / "data").resolve()
            if permitido in alvo.parents and alvo.is_file() and alvo.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".webp"):
                dados = alvo.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", f"image/{alvo.suffix.lstrip('.').replace('jpg', 'jpeg')}")
                self.send_header("Content-Length", str(len(dados)))
                self.end_headers()
                self.wfile.write(dados)
            else:
                self.send_error(404)
            return
        rel = "index.html" if path == "/" else path.lstrip("/")
        file = (WEB / rel).resolve()
        if WEB.resolve() not in file.parents and file != WEB.resolve():
            self.send_error(403)
            return
        if not file.is_file():
            self.send_error(404)
            return
        data = file.read_bytes()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(file.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                self._json(200, ollama.start_model())
                return
            if path == "/api/stop":
                self._json(200, ollama.stop_model())
                return
            if path == "/api/chat":
                body = self._read_json()
                messages = body.get("messages") or []
                if not messages:
                    self._json(400, {"error": "messages vazio"})
                    return
                text = ollama.chat(
                    messages,
                    stream=False,
                    use_file=bool(body.get("use_file", True)),
                    use_lora=bool(body.get("use_lora", True)),
                    modo=body.get("modo") or "conversa",
                )
                aviso = ""
                if body.get("use_file", True):
                    from agentepc.memory import read_knowledge

                    cabe, aviso = ollama.cabe_no_contexto(read_knowledge() or "")
                self._json(200, {"text": text, "aviso": aviso,
                                 "use_file": bool(body.get("use_file", True)), **ollama.status()})
                return
            if path == "/api/knowledge":
                body = self._read_json()
                if body.get("original"):
                    memory.save_original(body.get("name") or "nota.md", body["original"])
                texto = body.get("text") or ""
                if body.get("from_file"):
                    # arquivo grande: copia do disco, sem passar o conteudo pelo navegador
                    src = Path(body["from_file"])
                    texto = src.read_text(encoding="utf-8") if src.exists() else texto
                saved = memory.write_knowledge(body.get("name") or "nota.md", texto)
                self._json(200, {"ok": True, "file": saved.name, "files": memory.knowledge_index()})
                return
            if path == "/api/learn":
                body = self._read_json()
                if body.get("train"):
                    self._json(
                        200,
                        train.start_job(
                            force=bool(body.get("force")),
                            file=body.get("file") or None,
                            fresh=bool(body.get("fresh")),
                        ),
                    )
                else:
                    self._json(200, train.plan(force=bool(body.get("force")), file=body.get("file") or None))
                return
            if path == "/api/train-delete":
                b = self._read_json()
                if b.get("file"):
                    self._json(200, train.delete_group(b["file"]))
                else:
                    self._json(200, train.delete_training(b.get("id") or ""))
                return
            if path == "/api/knowledge-delete":
                self._json(200, memory.delete_knowledge(self._read_json().get("name") or ""))
                return
            if path == "/api/clean-cache":
                self._json(200, health.limpar_cache())
                return
            if path == "/api/model-plan":
                self._json(200, health.plano_troca(self._read_json().get("id") or ""))
                return
            if path == "/api/use-model":
                b = self._read_json()
                self._json(200, health.trocar_modelo(b.get("id") or "", bool(b.get("apagar", True))))
                return
            if path == "/api/prepare":
                b = self._read_json()
                self._json(200, provision.preparar(b, bool(b.get("executar"))))
                return
            if path == "/api/prepare-ollama":
                self._json(200, provision.apontar_ollama(self._read_json().get("host") or ""))
                return
            if path == "/api/crawl":
                b = self._read_json()
                self._json(200, crawler.descobrir(
                    b.get("url") or "", bool(b.get("browser", True)), bool(b.get("externos")),
                    int(b.get("pages") or 0), b.get("incluir") or "", b.get("excluir") or "",
                    bool(b.get("so_abaixo", True)), b.get("assunto") or "",
                ))
                return
            if path == "/api/doc-atualizar":
                self._json(200, catalogo_docs.atualizar())
                return
            if path == "/api/crawl-zip":
                b = self._read_json()
                self._json(200, crawler.importar_zip(b.get("origem") or "", b.get("assunto") or ""))
                return
            if path == "/api/crawl-build":
                b = self._read_json()
                self._json(200, crawler.montar(b.get("urls") or [], bool(b.get("images", True))))
                return
            if path == "/api/crawl-clear":
                self._json(200, crawler.limpar())
                return
            if path == "/api/crawl-stop":
                self._json(200, crawler.stop())
                return
            if path == "/api/export":
                b = self._read_json()
                self._json(200, export.start(b.get("id") or "", b.get("name") or "", bool(b.get("base")), bool(b.get("archive", True))))
                return
            if path == "/api/export-varios":
                b = self._read_json()
                self._json(200, export.start_varios(
                    b.get("ids") or [], b.get("name") or "", bool(b.get("base")),
                    bool(b.get("archive", True)), b.get("sistema") or "ubuntu"))
                return
            if path == "/api/export-delete":
                self._json(200, export.delete(self._read_json().get("name") or ""))
                return
            if path == "/api/adapter-use":
                train.set_active(self._read_json().get("id") or "")
                self._json(200, train.adapters())
                return
            if path == "/api/train-test":
                self._json(200, train.test_group(self._read_json().get("file") or ""))
                return
            if path == "/api/format":
                b = self._read_json()
                self._json(200, formatter.start(b.get("raw") or "", b.get("subject") or "",
                                                b.get("source_file") or "", b.get("lote") or ""))
                return
            if path == "/api/format-stop":
                self._json(200, formatter.stop())
                return
            if path == "/api/fix":
                b = self._read_json()
                self._json(200, formatter.corrigir(b.get("text") or "", b.get("subject") or ""))
                return
            if path == "/api/coletar":
                b = self._read_json()
                self._json(200, coletor.coletar(
                    b.get("termo") or "", int(b.get("quantidade") or 24),
                    b.get("licenca") or "livres", b.get("fontes") or "openverse,wikimedia",
                    b.get("nome") or ""))
                return
            if path == "/api/coletar-stop":
                self._json(200, coletor.parar())
                return
            if path == "/api/colecao-delete":
                self._json(200, coletor.apagar(self._read_json().get("id") or ""))
                return
            if path == "/api/som-coletar":
                b = self._read_json()
                self._json(200, som.coletar(
                    b.get("termo") or "", int(b.get("quantidade") or 20),
                    b.get("licenca") or "livres", b.get("fontes") or "openverse,wikimedia",
                    b.get("nome") or ""))
                return
            if path == "/api/som-coletar-stop":
                self._json(200, som.parar_coleta())
                return
            if path == "/api/som-colecao-delete":
                self._json(200, som.apagar_colecao(self._read_json().get("id") or ""))
                return
            if path == "/api/som-importar":
                b = self._read_json()
                self._json(200, som.importar(b.get("conjunto") or "", b.get("classe") or "",
                                             b.get("colecao") or ""))
                return
            if path == "/api/som-treinar":
                b = self._read_json()
                self._json(200, som.treinar(b.get("conjunto") or "", int(b.get("epocas") or 8),
                                            b.get("nome") or "", float(b.get("limiar") or 0.9)))
                return
            if path == "/api/som-stop":
                self._json(200, som.parar())
                return
            if path == "/api/som-modelo-delete":
                self._json(200, som.apagar(self._read_json().get("id") or ""))
                return
            if path == "/api/som-gerar":
                b = self._read_json()
                self._json(200, som.gerar(b.get("pedido") or "", int(b.get("segundos") or 8),
                                          b.get("estilo") or ""))
                return
            if path == "/api/som-gerado-delete":
                self._json(200, som.apagar_gerado(self._read_json().get("id") or ""))
                return
            if path == "/api/som-estilo-treinar":
                b = self._read_json()
                self._json(200, som.treinar_geracao(
                    b.get("colecao") or "", int(b.get("passos") or 200),
                    b.get("nome") or "", b.get("gatilho") or ""))
                return
            if path == "/api/som-estilo-delete":
                self._json(200, som.apagar_estilo(self._read_json().get("id") or ""))
                return
            if path == "/api/som-ouvir":
                b = self._read_json()
                self._json(200, som.ouvir(b.get("id") or "", b.get("arquivo") or ""))
                return
            if path == "/api/tresd-gerar":
                b = self._read_json()
                self._json(200, tresd.gerar(b.get("pedido") or "", b.get("foto") or "",
                                            int(b.get("passos") or 64), float(b.get("guia") or 15.0),
                                            b.get("nome") or ""))
                return
            if path == "/api/tresd-delete":
                self._json(200, tresd.apagar(self._read_json().get("id") or ""))
                return
            if path == "/api/tresd-coletar":
                b = self._read_json()
                self._json(200, tresd.coletar(b.get("termo") or "", int(b.get("quantidade") or 6),
                                              b.get("nome") or ""))
                return
            if path == "/api/tresd-coletar-stop":
                self._json(200, tresd.parar_coleta())
                return
            if path == "/api/tresd-referencia-delete":
                self._json(200, tresd.apagar_referencia(self._read_json().get("id") or ""))
                return
            if path == "/api/instalar-capacidade":
                self._json(200, capacidades.instalar(self._read_json().get("id") or ""))
                return
            if path == "/api/aprovar":
                b = self._read_json()
                self._json(200, consolidado.aprovar(b.get("tipo") or "imagem", b.get("id") or "",
                                                    bool(b.get("aprovado")), b.get("nota") or ""))
                return
            if path == "/api/gerar":
                b = self._read_json()
                self._json(200, imagem.gerar(b.get("pedido") or "", b.get("estilo") or "",
                                             int(b.get("passos") or 25), float(b.get("forca") or 0.9)))
                return
            if path == "/api/imagem-treinar":
                b = self._read_json()
                self._json(200, imagem.treinar(
                    b.get("colecao") or "", b.get("gatilho") or "", int(b.get("passos") or 600),
                    b.get("nome") or "", int(b.get("resolucao") or 512),
                    tipo=b.get("tipo") or "estilo", base=b.get("base") or "sd15"))
                return
            if path == "/api/classificador-treinar":
                b = self._read_json()
                self._json(200, classificador.treinar(
                    b.get("conjunto") or "", int(b.get("epocas") or 8), b.get("nome") or "",
                    float(b.get("limiar") or 0.9)))
                return
            if path == "/api/classificador-stop":
                self._json(200, classificador.parar())
                return
            if path == "/api/classificador-importar":
                b = self._read_json()
                self._json(200, classificador.importar(
                    b.get("conjunto") or "", b.get("classe") or "", b.get("colecao") or ""))
                return
            if path == "/api/classificador-delete":
                self._json(200, classificador.apagar(self._read_json().get("id") or ""))
                return
            if path == "/api/imagem-stop":
                self._json(200, imagem.parar())
                return
            if path == "/api/imagem-delete":
                self._json(200, imagem.apagar(self._read_json().get("id") or ""))
                return
            if path == "/api/lote-delete":
                self._json(200, lotes.apagar(self._read_json().get("id") or ""))
                return
            if path == "/api/lint":
                self._json(200, {"warnings": formatter.lint(self._read_json().get("text") or "")})
                return
            if path == "/api/train-stop":
                self._json(200, train.stop_job())
                return
            if path == "/api/teach":
                body = self._read_json()
                fact = (body.get("text") or "").strip()
                path_saved = memory.append_profile(fact)
                self._json(
                    200,
                    {
                        "ok": True,
                        "file": str(path_saved),
                        "profile": memory.read_profile(),
                        "note": "gravado no caderno. pesos intactos.",
                    },
                )
                return
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self.send_error(404)


def serve(host: str = "0.0.0.0", port: int = 8765) -> None:
    ollama.ensure_server()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"no Chrome do Windows abra:  http://127.0.0.1:{port}")
    print("nao use /web/index.html e nao omita a porta")
    print(f"modelo: {ollama.model_name()}  — desligar nao apaga os pesos")
    httpd.serve_forever()
