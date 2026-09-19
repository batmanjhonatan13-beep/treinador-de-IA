"""Empacota o conhecimento treinado para rodar em outra maquina.

Nao funde nada: sai o Qwem original (ou a instrucao de baixa-lo) + o adaptador do
treino + um script que junta os dois na hora de responder. Quem receber o pacote roda
um comando e tem o modelo com aquele conhecimento.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from agentepc import sistemas, train
from agentepc.config import load, resolve
from agentepc.prompting import SYSTEM, prompt_for

RUN_PY = '''"""Roda o modelo com o conhecimento treinado. Nao precisa de internet se a pasta base/ veio junto.

    python run.py "Qual e o nome da mae de Carla?"      # uma pergunta
    python run.py                                        # conversa no terminal
    python run.py --serve 8000                           # HTTP: POST /chat {"pergunta": "..."}
"""

import json
import os
import sys
from pathlib import Path

# Em maquina sem compilador C, o torch tenta compilar kernels no 1o passo e quebra.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "adapter"
LOCAL_BASE = HERE / "base"
BASE = str(LOCAL_BASE) if (LOCAL_BASE / "config.json").exists() else "%(base_hf)s"
SYSTEM = %(system)r

_bundle = None


def load_model():
    """4 bits e o padrao de proposito: o adaptador foi treinado sobre a base nesse formato.
    Carregar em precisao cheia muda as respostas (medido no projeto que gerou este pacote)."""
    global _bundle
    if _bundle:
        return _bundle
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ADAPTER) if (ADAPTER / "tokenizer.json").exists() else BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {}
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            kwargs["device_map"] = "auto"
        except ImportError:
            print("[aviso] bitsandbytes ausente: rodando em 16 bits, as respostas podem sair diferentes do treino")
            kwargs["dtype"] = torch.float16
            kwargs["device_map"] = "auto"
    else:
        print("[aviso] sem GPU: rodando na CPU em 32 bits. Funciona, mas e lento e pode variar do treino")
    model = AutoModelForCausalLM.from_pretrained(BASE, **kwargs)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    _bundle = (tok, model)
    return _bundle


def ask(pergunta, max_new_tokens=64):
    import re

    import torch

    tok, model = load_model()
    prompt = f"system: {SYSTEM}\\nuser: {pergunta}\\nassistant:"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    texto = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.split(r"\\n\\s*(?:user|assistant|system)\\s*:", texto)[0].strip()


def serve(port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or "{}")
            texto = ask(body.get("pergunta") or body.get("prompt") or "")
            data = json.dumps({"resposta": texto}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"POST http://127.0.0.1:{port}/chat  {{\\"pergunta\\": \\"...\\"}}")
    HTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--serve":
        serve(int(args[1]) if len(args) > 1 else 8000)
    elif args:
        print(ask(" ".join(args)))
    else:
        print("Pergunte (Ctrl+C para sair).")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                print(ask(q))
'''

REQS = "torch\ntransformers>=4.44\npeft\naccelerate\nbitsandbytes\n"

INSTALL_SH = '''#!/usr/bin/env bash
# Prepara o ambiente para %(sistema)s e faz uma pergunta de teste.
set -e
cd "$(dirname "$0")"
SUDO=""
[ "$(id -u)" = 0 ] || { command -v sudo >/dev/null && SUDO="sudo"; }

%(basico)s
PY=${PY:-python3}
$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo
echo "Teste:"
.venv/bin/python run.py %(sample)s
'''

INSTALL_PS1 = '''# Prepara o ambiente no Windows e faz uma pergunta de teste.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command python -EA 0)) {
  if (Get-Command winget -EA 0) { winget install --silent --accept-package-agreements --accept-source-agreements --id Python.Python.3.12 }
  else { throw "instale o Python 3 e rode de novo" }
}
python -m venv .venv
.\\.venv\\Scripts\\pip.exe install --upgrade pip
.\\.venv\\Scripts\\pip.exe install -r requirements.txt
Write-Output ""
Write-Output "Teste:"
.\\.venv\\Scripts\\python.exe run.py %(sample)s
'''


def _jobs_dir() -> Path:
    path = resolve("data/export")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base_snapshot() -> Path | None:
    """Pasta do Qwen no cache do Hugging Face, para copiar junto quando pedido."""
    repo = load()["model"]["base_hf"].replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}" / "snapshots"
    if not root.exists():
        return None
    snaps = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


_job: dict = {"state": "idle", "lines": [], "name": "", "path": ""}


def _say(msg: str) -> None:
    _job["lines"].append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "msg": msg})


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def listing() -> list[dict]:
    out = []
    for path in sorted(_jobs_dir().iterdir()) if _jobs_dir().exists() else []:
        if path.is_dir():
            meta = path / "pacote.json"
            info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            out.append({"name": path.name, "path": str(path), "bytes": _size(path), **info})
        elif path.suffix == ".gz":
            out.append({"name": path.name, "path": str(path), "bytes": path.stat().st_size, "archive": True})
    return out


def prontos() -> list[dict]:
    """So aparece no export o treino que fechou 100% na prova sem consulta."""
    keys = ("id", "ts", "file", "facts", "passed", "total", "tested", "size", "base_hf", "modelo")
    return [{k: r.get(k) for k in keys} for r in train.history(100) if r.get("consolidated") and r.get("snapshot")]


def status() -> dict:
    return {**_job, "items": listing(), "runs": prontos(), "base_local": _base_snapshot() is not None}


def start(run_id: str, name: str = "", include_base: bool = False, archive: bool = True,
          sistema: str = "ubuntu") -> dict:
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um pacote sendo montado"}
    if train.job_status()["state"] == "running":
        return {"accepted": False, "reason": "treino em andamento; espere ou pare"}
    rows = {r["id"]: r for r in train._rows() if r.get("snapshot")}
    row = rows.get(run_id)
    if not row:
        return {"accepted": False, "reason": "esse treino nao tem pesos guardados"}
    if not row.get("consolidated"):
        return {
            "accepted": False,
            "reason": "so exporto conhecimento consolidado: o modelo tem que acertar a prova inteira sem consulta. "
            "Rode 'Testar consolidacao' ou treine mais esse arquivo.",
        }
    name = re.sub(r"[^a-z0-9_-]+", "-", (name or f"modelo-{run_id}").lower()).strip("-") or f"modelo-{run_id}"
    dest = _jobs_dir() / name
    if dest.exists():
        return {"accepted": False, "reason": f"ja existe um pacote chamado {name}"}
    if not sistemas.por_id(sistema):
        return {"accepted": False, "reason": "sistema desconhecido"}
    _job.update({"state": "running", "lines": [], "name": name, "path": str(dest)})

    def _go() -> None:
        try:
            cfg = load()
            src = train._versions_dir() / run_id
            dest.mkdir(parents=True)
            _say(f"copiando o conhecimento treinado ({row['file']})")
            shutil.copytree(src, dest / "adapter", ignore=shutil.ignore_patterns("runs"))
            sample = "Quem e o dono deste modelo?"
            try:
                from agentepc import exam

                quiz = exam.build(row["file"])
                if quiz:
                    sample = quiz[0]["q"]
            except Exception:
                pass
            # o adaptador so funciona no modelo em que foi treinado
            base_treino = row.get("base_hf") or cfg["model"]["base_hf"]
            if base_treino != cfg["model"]["base_hf"]:
                _say(f"atencao: este conhecimento foi treinado em {base_treino}, "
                     f"diferente do modelo atual ({cfg['model']['base_hf']}). O pacote leva o de origem.")
            _say(f"modelo de origem: {base_treino}")
            (dest / "run.py").write_text(
                RUN_PY % {"base_hf": base_treino, "system": SYSTEM}, encoding="utf-8"
            )
            (dest / "requirements.txt").write_text(REQS, encoding="utf-8")
            info_sis = sistemas.por_id(sistema)
            amostra = json.dumps(sample, ensure_ascii=False)
            if sistema == "windows":
                (dest / "instalar.ps1").write_text(INSTALL_PS1 % {"sample": amostra}, encoding="utf-8")
            else:
                sh = dest / "instalar.sh"
                sh.write_text(
                    INSTALL_SH % {
                        "sample": amostra,
                        "sistema": info_sis["nome"],
                        "basico": sistemas.bloco_basico(sistema),
                    },
                    encoding="utf-8",
                )
                sh.chmod(0o755)
            _say(f"instalador para {info_sis['nome']}")
            if include_base:
                snap = _base_snapshot()
                if not snap:
                    raise RuntimeError("o Qwen nao esta no cache local; treine ou baixe uma vez antes de incluir a base")
                _say("copiando o Qwen original (alguns GB, demora)")
                shutil.copytree(snap, dest / "base", symlinks=False, ignore=shutil.ignore_patterns("*.pth", ".*"))
            meta = {
                "run_id": run_id,
                "file": row["file"],
                "facts": row.get("facts"),
                "prova": row.get("tested") or [row.get("passed"), row.get("total")],
                "consolidado": bool(row.get("consolidated")),
                "base_hf": base_treino,
                "modelo_ollama": row.get("modelo"),
                "base_incluida": include_base,
                "sistema": sistema,
                "sistema_nome": (sistemas.por_id(sistema) or {}).get("nome", sistema),
                "criado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prompt": prompt_for("SUA PERGUNTA"),
            }
            (dest / "pacote.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            (dest / "README.md").write_text(_readme(meta, name, sample), encoding="utf-8")
            _say(f"pacote montado ({_size(dest) / 1e6:.0f} MB)")
            if archive:
                _say("compactando para upload")
                tgz = _jobs_dir() / f"{name}.tar.gz"
                with tarfile.open(tgz, "w:gz") as tar:
                    tar.add(dest, arcname=name)
                _say(f"arquivo pronto: {tgz} ({tgz.stat().st_size / 1e6:.0f} MB)")
            _job["state"] = "done"
            _say("copie a pasta (ou o .tar.gz) para o outro servidor e rode ./instalar.sh")
        except Exception as exc:
            shutil.rmtree(dest, ignore_errors=True)
            _job["state"] = "error"
            _say(f"erro: {exc}")

    threading.Thread(target=_go, name="export", daemon=True).start()
    return {"accepted": True, "name": name}


def delete(name: str) -> dict:
    if _job["state"] == "running":
        return {"ok": False, "reason": "pacote sendo montado"}
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "", name)
    for path in (_jobs_dir() / safe, _jobs_dir() / f"{safe}.tar.gz"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    return {"ok": True, "items": listing()}


def _readme(meta: dict, name: str, sample: str) -> str:
    win = meta.get("sistema") == "windows"
    rodar = (
        f'''```powershell
.\\instalar.ps1
.\\.venv\\Scripts\\python.exe run.py "{sample}"
.\\.venv\\Scripts\\python.exe run.py --serve 8000
```'''
        if win else
        f'''```bash
./instalar.sh
.venv/bin/python run.py "{sample}"
.venv/bin/python run.py --serve 8000
```'''
    )
    base_line = (
        "A pasta `base/` ja traz o Qwen: nao precisa de internet."
        if meta["base_incluida"]
        else f"O `run.py` baixa o `{meta['base_hf']}` do Hugging Face na primeira vez (precisa de internet uma vez)."
    )
    return f"""# {name}

Qwen original + o conhecimento treinado em `{meta['file']}`.

- conhecimento: **{meta['facts']} linhas**, prova sem consulta **{meta['prova'][0]}/{meta['prova'][1]}**{' (consolidado)' if meta['consolidado'] else ''}
- **modelo de origem: `{meta['base_hf']}`** — o conhecimento só funciona neste modelo
- gerado em {meta['criado']}
- preparado para **{meta.get('sistema_nome', '?')}**

## Rodar

{rodar}

O instalador já traz os pacotes de sistema certos para {meta.get('sistema_nome', 'este sistema')}.

{base_line}

## Como funciona

`adapter/` sao os pesos do treino (LoRA). O `run.py` carrega **`{meta['base_hf']}`**, aplica o
adaptador por cima e responde. O modelo base nao foi alterado.

**O adaptador so serve nesse modelo.** Ele aprendeu ajustando aquelas camadas, naquele
tamanho; em outro modelo as formas nem batem, e quando batem a resposta vira ruido. Se
quiser outro modelo base, treine o arquivo de novo nele.

**Carregue em 4 bits.** O adaptador foi treinado sobre a base quantizada em 4 bits; em
precisao cheia as mesmas perguntas saem diferentes. O `run.py` ja faz isso quando ha GPU
e avisa quando nao da.

O prompt tem que ser exatamente este, senao o conhecimento nao e acionado:

```
{meta['prompt']}
```
"""
