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


# O chat servia a dois donos com um prompt so, e o resultado era ruim para os dois: pedir
# "me ensine" e receber uma linha. Agora o modo e explicito.
#   curto    — uma linha, so o fato. E o formato do treino e das provas.
#   conversa — explica, ensina, leva em conta o que ja foi dito.
def system_prompt(use_file: bool, modo: str = "conversa") -> str:
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
    if modo == "conversa":
        return (
            "Voce e Qwen, um modelo local rodando no PC do usuario. Responda em portugues.\n"
            "Leve em conta a conversa inteira, nao so a ultima mensagem: se o usuario disser "
            "'me ensine' ou 'explique melhor', ele esta falando do assunto anterior.\n"
            "Quando pedirem para ensinar ou explicar, explique de verdade: passo a passo, com "
            "exemplo, quantas linhas forem precisas.\n"
            "Se nao souber, diga que nao sabe — mas nao use isso para fugir de uma explicacao.\n"
            "Nao mencione arquivo, caderno, LoRA, consulta nem sistema."
        )
    return (
        "Voce e Qwen. Responda em portugues, curto, so com o que voce ja sabe. "
        "Se nao souber, diga que nao sabe, com as suas palavras. "
        "Nao mencione arquivo, caderno, LoRA, consulta nem sistema."
    )


# A janela de contexto do Ollama vem em 4096 fichas por padrao. Um caderno maior que isso
# e CORTADO EM SILENCIO: o modelo responde "nao sei" com o fato ali, no pedaco que foi
# descartado. Medido aqui: 50 mil caracteres com 4096 -> resposta errada em 1,7 s; os
# mesmos 50 mil com 16384 -> resposta certa em 7 s. Entao a janela passa a acompanhar o
# tamanho do caderno, e o que nao couber e dito em voz alta.
JANELAS = (4096, 8192, 16384, 32768)
FICHAS_POR_CARACTERE = 1 / 3.5


def janela_para(texto: str) -> int:
    fichas = len(texto) * FICHAS_POR_CARACTERE + 700     # folga para a pergunta e a resposta
    for tamanho in JANELAS:
        if fichas < tamanho * 0.9:
            return tamanho
    return JANELAS[-1]


def cabe_no_contexto(texto: str) -> tuple[bool, str]:
    """Diz se o caderno inteiro cabe, e o que fazer quando nao cabe."""
    fichas = len(texto) * FICHAS_POR_CARACTERE
    teto = JANELAS[-1] * 0.9
    if fichas <= teto:
        return True, ""
    sobra = int((fichas - teto) * 3.5)
    return False, (
        f"o caderno tem {len(texto)//1000} mil caracteres e so cabem cerca de "
        f"{int(teto * 3.5)//1000} mil na janela do modelo: {sobra//1000} mil ficam de fora "
        "e o modelo nem sabe que existiram. Divida o arquivo, ou treine esse conteudo em "
        "vez de consultar."
    )


def with_profile(messages: list[dict], use_file: bool = True, modo: str = "conversa") -> list[dict]:
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt(use_file, modo)}, *rest]


def chat(messages: list[dict], stream: bool = False, use_file: bool = True,
         use_lora: bool = True, modo: str = "conversa"):
    if not use_file and use_lora:
        adapter = serving_adapter()
        if adapter:
            try:
                from agentepc.engine import chat_adapter
                from agentepc.prompting import SYSTEM

                # Um conhecimento treinado so foi visto num formato: uma pergunta, uma
                # resposta, com esta linha de sistema. Jogar a conversa inteira nesse molde
                # e pedir o que ele nunca viu — e a resposta sai quebrada. Entao, com o
                # conhecimento ligado, vale a ultima pergunta. Esta escrito na tela.
                ultima = next((m.get("content", "") for m in reversed(messages)
                               if m.get("role") == "user"), "")
                return chat_adapter([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": ultima}], adapter=adapter)
            except Exception as exc:
                print(f"[lora] fallback ollama: {exc}")
    ensure_server()
    montadas = with_profile(messages, use_file=use_file, modo=modo)
    inteiro = "".join(m.get("content", "") for m in montadas)
    payload = {
        "model": model_name(),
        "messages": montadas,
        "stream": stream,
        "keep_alive": "30m",
        "options": {"num_ctx": janela_para(inteiro)},
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
