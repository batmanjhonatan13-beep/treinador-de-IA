from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agentepc import crawler, export, formatter, health, memory, ollama, provision, sistemas, train
from agentepc.config import ROOT

WEB = ROOT / "web"


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
                )
                self._json(200, {"text": text, "use_file": bool(body.get("use_file", True)), **ollama.status()})
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
                self._json(200, crawler.start(
                    b.get("url") or "", int(b.get("pages") or 0),
                    bool(b.get("browser", True)), bool(b.get("images", True)),
                    bool(b.get("externos")),
                ))
                return
            if path == "/api/crawl-stop":
                self._json(200, crawler.stop())
                return
            if path == "/api/export":
                b = self._read_json()
                self._json(200, export.start(b.get("id") or "", b.get("name") or "", bool(b.get("base")), bool(b.get("archive", True))))
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
                self._json(200, formatter.start(b.get("raw") or "", b.get("subject") or "", b.get("source_file") or ""))
                return
            if path == "/api/format-stop":
                self._json(200, formatter.stop())
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
