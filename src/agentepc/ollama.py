from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentepc.config import load, resolve
from agentepc.memory import knowledge_index, read_knowledge, read_profile
from agentepc.train import adapter_exists, serving_adapter

DEFAULT_HOST = "127.0.0.1:11434"
LOG = Path("/tmp/ollama-serve.log")


def host() -> str:
    cfg = load().get("model") or {}
    return os.environ.get("OLLAMA_HOST") or cfg.get("ollama_host") or DEFAULT_HOST


def base_name() -> str:
    """Modelo original do config. Usado para formatar dados e gerar perguntas, nunca o fundido."""
    return (load().get("model") or {}).get("name") or "qwen2.5:3b"


def model_name() -> str:
    return base_name()


def ollama_bin() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "ollama"
    if local.exists():
        return str(local)
    raise FileNotFoundError("ollama nao encontrado em ~/.local/bin")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".local" / "bin") + os.pathsep + env.get("PATH", "")
    lib = Path.home() / ".local" / "lib" / "ollama"
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["OLLAMA_HOST"] = host()
    return env


def _url(path: str) -> str:
    h = host()
    if not h.startswith("http"):
        h = "http://" + h
    return h.rstrip("/") + path


def request(path: str, payload: dict | None = None, timeout: float = 120) -> dict | list:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_url(path), data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"ollama indisponivel em {_url(path)}: {exc}") from exc
    return json.loads(raw) if raw else {}


def ping() -> bool:
    try:
        request("/api/tags", timeout=2)
        return True
    except RuntimeError:
        return False


def _is_local_host(h: str) -> bool:
    h = h.replace("http://", "").replace("https://", "").split(":")[0]
    return h in {"127.0.0.1", "localhost", "::1"}


def ensure_server() -> None:
    if ping():
        return
    if not _is_local_host(host()):
        for _ in range(60):
            if ping():
                return
            time.sleep(1)
        raise RuntimeError(f"ollama remoto nao respondeu em {host()}")
    LOG.write_text("", encoding="utf-8")
    subprocess.Popen(
        [ollama_bin(), "serve"],
        env=_env(),
        stdout=LOG.open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(40):
        if ping():
            return
        time.sleep(0.25)
    raise RuntimeError("ollama serve nao subiu. veja /tmp/ollama-serve.log")


def tags() -> list[str]:
    data = request("/api/tags", timeout=5)
    return [m.get("name", "") for m in data.get("models") or []]


def loaded() -> list[str]:
    try:
        data = request("/api/ps", timeout=5)
    except RuntimeError:
        return []
    return [m.get("name", "") for m in data.get("models") or []]


def model_ready() -> bool:
    name = model_name()
    return any(name == t or t.startswith(name + ":") or name.startswith(t) for t in tags())


def start_model() -> dict:
    ensure_server()
    name = model_name()
    if not model_ready():
        raise RuntimeError(f"modelo {name} nao esta no disco. rode: ollama pull {name}")
    request(
        "/api/generate",
        {"model": name, "prompt": "ok", "stream": False, "keep_alive": "30m"},
        timeout=180,
    )
    return status()


def stop_model() -> dict:
    if not ping():
        return status()
    name = model_name()
    try:
        request(
            "/api/generate",
            {"model": name, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=30,
        )
    except RuntimeError:
        subprocess.run([ollama_bin(), "stop", name], env=_env(), check=False)
    return status()


def status() -> dict:
    up = ping()
    names = loaded() if up else []
    name = model_name()
    on = any(name in n or n.startswith(name.split(":")[0]) for n in names)
    return {
        "server": up,
        "model": name,
        "on": on,
        "loaded": names,
        "on_disk": model_ready() if up else False,
        "profile": bool(read_profile()),
        "files": knowledge_index(),
        "lora": serving_adapter() is not None,
        "note": "Arquivo e opcional na consulta. LoRA e o peso assado.",
    }


def system_prompt(use_file: bool) -> str:
    base = (
        "Voce e um modelo local (Qwen) rodando no PC do usuario via Ollama.\n"
        "Voce NAO esta na Alibaba Cloud, nem na internet, nem em API paga.\n"
        "Responda em portugues, curto e direto.\n"
    )
    if use_file:
        knowledge = read_knowledge() or "(caderno vazio)"
        return (
            "Voce e o assistente Qwen no PC do usuario. Voce NAO e o Jhonatan.\n"
            "O caderno abaixo descreve o USUARIO, Jhonatan Santos De Menezes.\n"
            "Se perguntarem quem e o Jhonatan, o pai, a mae, o irmao ou a irma, "
            "responda com os fatos do caderno, em terceira pessoa.\n"
            "Nao diga que o caderno esta vazio se o fato estiver listado.\n"
            "Nao diga 'nao sou o Jhonatan' no lugar da resposta. So diga que nao sabe "
            "se o fato realmente nao estiver no caderno.\n\n"
            f"{knowledge}"
        )
    return (
        "Voce e Qwen. Responda em portugues, curto, so com o que voce ja sabe. "
        "Se nao souber, diga que nao sabe, com as suas palavras. "
        "Nao mencione arquivo, caderno, LoRA, consulta nem sistema."
    )


def with_profile(messages: list[dict], use_file: bool = True) -> list[dict]:
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt(use_file)}, *rest]


def chat(messages: list[dict], stream: bool = False, use_file: bool = True, use_lora: bool = True):
    if not use_file and use_lora:
        adapter = serving_adapter()
        if adapter:
            try:
                from agentepc.engine import chat_adapter

                return chat_adapter(with_profile(messages, use_file=False), adapter=adapter)
            except Exception as exc:
                print(f"[lora] fallback ollama: {exc}")
    ensure_server()
    payload = {
        "model": model_name(),
        "messages": with_profile(messages, use_file=use_file),
        "stream": stream,
        "keep_alive": "30m",
    }
    if not stream:
        data = request("/api/chat", payload, timeout=300)
        return (data.get("message") or {}).get("content") or ""
    req = urllib.request.Request(
        _url("/api/chat"),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=300)
