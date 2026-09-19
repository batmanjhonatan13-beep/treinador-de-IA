"""Empacota o conhecimento treinado para rodar em outra maquina.

Nao funde nada: sai o Qwem original (ou a instrucao de baixa-lo) + o adaptador do
treino + um script que junta os dois na hora de responder. Quem receber o pacote roda
um comando e tem o modelo com aquele conhecimento.
"""

from __future__ import annotations

import json
import re
import shlex
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _say(msg: str) -> None:
    _job["lines"].append({"ts": _now(), "msg": msg})


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
    """So entra no pacote o que esta consolidado — cada tipo pela sua regra."""
    from agentepc import consolidado

    keys = ("id", "ts", "file", "facts", "passed", "total", "tested", "size", "base_hf", "modelo")
    saida = [{**{k: r.get(k) for k in keys}, "tipo": "texto"}
             for r in train.history(100) if r.get("consolidated") and r.get("snapshot")]
    for item in consolidado.consolidados():
        if item["tipo"] == "texto":
            continue
        saida.append({"id": item["id"], "tipo": item["tipo"], "file": item["nome"],
                      "ts": "", "facts": None, "passed": None, "total": None,
                      "modelo": item.get("modelo", ""), "detalhe": item["detalhe"]})
    return saida


def status() -> dict:
    return {**_job, "items": listing(), "runs": prontos(), "base_local": _base_snapshot() is not None}


_MAIN_SOM = '\ndef main(caminho: str) -> None:\n    dados = torch.load(Path(__file__).parent / "modelo" / "modelo.pt",\n                       map_location="cpu", weights_only=False)\n    rede = models.resnet18()\n    rede.fc = torch.nn.Linear(rede.fc.in_features, len(dados["classes"]))\n    rede.load_state_dict(dados["pesos"])\n    rede.eval()\n    votos = torch.zeros(len(dados["classes"]))\n    pedacos = _fatias(Path(caminho), maximo=10)\n    if not pedacos:\n        raise SystemExit("audio curto demais ou em silencio")\n    with torch.no_grad():\n        for onda in pedacos:\n            mel = espectro(onda).unsqueeze(0).unsqueeze(0)\n            mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode="bilinear",\n                                                  align_corners=False)[0].repeat(3, 1, 1)\n            votos += torch.softmax(rede(mel.unsqueeze(0))[0], dim=0)\n    votos = votos / votos.sum()\n    ordem = votos.argsort(descending=True)\n    print(dados["classes"][int(ordem[0])], f"({float(votos[ordem[0]]):.0%})")\n    for i in ordem:\n        print(f"  {dados[\'classes\'][int(i)]}: {float(votos[i]):.0%}")\n\n\nif __name__ == "__main__":\n    if len(sys.argv) < 2:\n        raise SystemExit("uso: python usar.py audio.wav")\n    main(sys.argv[1])\n'


def _script_som() -> str:
    """Monta um `usar.py` que roda sozinho, sem o projeto do lado.

    O preparo do audio (taxa, tamanho do pedaco, espectrograma) tem que ser IGUAL ao do
    treino, senao o acerto cai sem dar erro nenhum. Em vez de pedir para copiar codigo a
    mao, o script sai pronto — e sai do proprio som.py, entao nao envelhece.
    """
    import inspect

    from agentepc import som

    partes = [
        '"""Classifica um audio com o modelo deste pacote:  python usar.py audio.wav"""',
        "import sys",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import soundfile as sf",
        "import torch",
        "from torchvision import models",
        "",
        f"SR = {som.SR}",
        f"JANELA = {som.JANELA}",
        f"MAX_SEG = {som.MAX_SEG}",
        "",
    ]
    for funcao in (som.carregar, som._filtros_mel, som.espectro, som._fatias):
        partes.append(inspect.getsource(funcao))
        partes.append("")
    partes.append(_MAIN_SOM)
    return "\n".join(partes)


def _script_som_estilo(item: dict) -> str:
    """Gera som com o LoRA de estilo deste pacote, sem precisar do projeto."""
    base = item.get("modelo") or "facebook/musicgen-small"
    return f'''"""Gera som com o estilo treinado:  python usar.py "gentle piano melody" 8"""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenForConditionalGeneration

AQUI = Path(__file__).parent
BASE = "{base}"


def main(pedido: str, segundos: int) -> None:
    treino = json.loads((AQUI / "modelo" / "treino.json").read_text(encoding="utf-8"))
    print("gatilho do treino:", treino.get("gatilho"))
    proc = AutoProcessor.from_pretrained(BASE)
    modelo = MusicgenForConditionalGeneration.from_pretrained(BASE)
    modelo.decoder = PeftModel.from_pretrained(modelo.decoder, str(AQUI / "modelo"))
    if torch.cuda.is_available():
        modelo = modelo.to("cuda")
    entrada = proc(text=[pedido], padding=True, return_tensors="pt").to(modelo.device)
    onda = modelo.generate(**entrada, do_sample=True, guidance_scale=3.0,
                           max_new_tokens=int(segundos) * 50)
    sf.write("saida.wav", onda[0, 0].float().cpu().numpy(),
             modelo.config.audio_encoder.sampling_rate)
    print("saida.wav pronto")


if __name__ == "__main__":
    pedido = sys.argv[1] if len(sys.argv) > 1 else "gentle piano melody"
    main(pedido, sys.argv[2] if len(sys.argv) > 2 else 8)
'''


def _empacota_outro(item: dict, dest: Path, sistema: str) -> dict:
    """Pacote de um treino que nao e de texto: os pesos + como usar."""
    from agentepc import classificador, imagem, sistemas, som, tresd

    if item["tipo"] == "imagem":
        origem = imagem.saida_raiz() / item["id"]
        leia = f"""# {item['id']}

Estilo treinado (LoRA) para **{item.get('modelo') or 'Stable Diffusion'}**.

O arquivo `pytorch_lora_weights.safetensors` entra no Automatic1111 (pasta `models/Lora`),
no ComfyUI ou em qualquer lugar que aceite LoRA de fora. Chame pelo gatilho anotado em
`treino.json`.

Consolidado por julgamento seu: {item['detalhe']}
"""
    elif item["tipo"] == "classificador":
        origem = classificador.saida_raiz() / item["id"]
        leia = f"""# {item['id']}

Classificador de imagem (resnet18). `modelo.pt` traz os pesos e a lista de categorias.

```python
import torch
from torchvision import models, transforms
from PIL import Image

d = torch.load("modelo.pt", map_location="cpu")
m = models.resnet18()
m.fc = torch.nn.Linear(m.fc.in_features, len(d["classes"]))
m.load_state_dict(d["pesos"]); m.eval()

prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
x = prep(Image.open("foto.jpg").convert("RGB")).unsqueeze(0)
print(d["classes"][m(x).argmax(1).item()])
```

Prova: {item['detalhe']}
"""
    elif item["tipo"] == "som-classificador":
        origem = som.modelos_raiz() / item["id"]
        leia = f"""# {item['id']}

Classificador de som (resnet18 sobre espectrograma).

    pip install torch torchvision soundfile scipy numpy
    python usar.py meu-audio.wav

O `usar.py` vai junto e nao depende do projeto: ele carrega o mesmo preparo de audio do
treino (mono, {som.SR} Hz, pedacos de {som.JANELA:.0f} s). Preparo diferente derruba o
acerto sem dar erro nenhum — por isso vai pronto em vez de vir explicado.

Prova: {item['detalhe']}
"""
    elif item["tipo"] == "som-estilo":
        origem = resolve("data/som-lora") / item["id"]
        leia = f"""# {item['id']}

Estilo de som (LoRA) para **{item.get('modelo') or 'facebook/musicgen-small'}**.
O gatilho esta em `modelo/treino.json`. Use o `usar.py` deste pacote.

ATENCAO A LICENCA: o MusicGen e CC-BY-NC. O LoRA aqui dentro e seu, mas so roda em cima
desse modelo, e o audio gerado por ele nao pode ser vendido.

Consolidado por julgamento seu: {item['detalhe']}
"""
    elif item["tipo"] == "3d":
        origem = tresd.raiz() / item["id"]
        leia = f"""# {item['id']}

Malha 3D gerada pelo Shap-E (OpenAI, Apache-2.0 — uso comercial liberado).

* `modelo/modelo.glb` — arraste para Blender, Unity, Unreal ou Godot
* `modelo/modelo.obj` — formato de texto, abre em qualquer lugar
* `modelo/modelo.ply` — original, com cor por vertice
* `modelo/previa.png` — como ficou

Nao ha pesos aqui: a malha e o produto.

Consolidado por julgamento seu: {item['detalhe']}
"""
    else:
        raise RuntimeError("tipo sem empacotamento")
    if not origem.is_dir():
        raise RuntimeError("os arquivos desse treino nao foram encontrados")
    shutil.copytree(origem, dest / "modelo", ignore=shutil.ignore_patterns("runs"))
    if item["tipo"] == "som-classificador":
        (dest / "usar.py").write_text(_script_som(), encoding="utf-8")
    if item["tipo"] == "som-estilo":
        (dest / "usar.py").write_text(_script_som_estilo(item), encoding="utf-8")
    (dest / "README.md").write_text(leia, encoding="utf-8")
    (dest / "pacote.json").write_text(json.dumps({
        "tipo": item["tipo"], "id": item["id"], "detalhe": item["detalhe"],
        "modelo": item.get("modelo", ""), "sistema": sistema,
        "sistema_nome": (sistemas.por_id(sistema) or {}).get("nome", sistema),
        "criado": _now(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


def _empacota_texto(run_id: str, row: dict, dest: Path, name: str, sistema: str,
                    include_base: bool) -> None:
    """Monta a pasta de um conhecimento de texto: adaptador + run.py + instalador.

    Sai daqui separado do resto porque o mesmo bloco serve para o pacote de um
    conhecimento so e para o pacote que junta varios.
    """
    cfg = load()
    src = train._versions_dir() / run_id
    dest.mkdir(parents=True, exist_ok=True)
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


def start(run_id: str, name: str = "", include_base: bool = False, archive: bool = True,
          sistema: str = "ubuntu") -> dict:
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um pacote sendo montado"}
    if train.job_status()["state"] == "running":
        return {"accepted": False, "reason": "treino em andamento; espere ou pare"}
    outros = {i["id"]: i for i in prontos() if i.get("tipo") != "texto"}
    rows = {r["id"]: r for r in train._rows() if r.get("snapshot")}
    row = rows.get(run_id)
    if not row and run_id not in outros:
        return {"accepted": False, "reason": "esse treino nao tem pesos guardados"}
    if not row:
        item = outros[run_id]
        name = re.sub(r"[^a-z0-9_-]+", "-", (name or item["id"]).lower()).strip("-")
        dest = _jobs_dir() / name
        if dest.exists():
            return {"accepted": False, "reason": f"ja existe um pacote chamado {name}"}
        _job.update({"state": "running", "lines": [], "name": name, "path": str(dest)})

        def _outro() -> None:
            try:
                dest.mkdir(parents=True)
                _say(f"empacotando {item['tipo']}: {item['id']}")
                _empacota_outro(item, dest, sistema)
                if archive:
                    _say("compactando para upload")
                    with tarfile.open(_jobs_dir() / f"{name}.tar.gz", "w:gz") as tar:
                        tar.add(dest, arcname=name)
                _job["state"] = "done"
                _say(f"pronto ({_size(dest) / 1e6:.0f} MB) — copie a pasta para onde quiser")
            except Exception as exc:
                shutil.rmtree(dest, ignore_errors=True)
                _job["state"] = "error"
                _say(f"erro: {exc}")

        threading.Thread(target=_outro, name="export-outro", daemon=True).start()
        return {"accepted": True, "name": name}
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
            _empacota_texto(run_id, row, dest, name, sistema, include_base)
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



# o que cada tipo precisa que esteja instalado na maquina que recebe o pacote
PIP_POR_TIPO = {
    "texto": ["torch", "transformers>=4.44", "peft", "accelerate", "bitsandbytes"],
    "imagem": ["torch", "diffusers>=0.31", "peft", "transformers>=4.44", "safetensors"],
    "classificador": ["torch", "torchvision", "pillow"],
    "som-classificador": ["torch", "torchvision", "soundfile", "scipy", "numpy"],
    "som-estilo": ["torch", "transformers>=4.44", "peft", "soundfile", "sentencepiece"],
    "3d": [],
}

COMO_USAR = {
    "texto": "cd texto/{id} && ../../.venv/bin/python run.py \"sua pergunta\"",
    "imagem": "copie imagem/{id}/modelo/*.safetensors para a pasta Lora do seu Automatic1111/ComfyUI",
    "classificador": "cd classificador/{id} && ../../.venv/bin/python ... (veja o README de dentro)",
    "som-classificador": "cd som-classificador/{id} && ../../.venv/bin/python usar.py audio.wav",
    "som-estilo": "cd som-estilo/{id} && ../../.venv/bin/python usar.py \"o que voce quer ouvir\" 8",
    "3d": "abra 3d/{id}/modelo/modelo.glb no Blender ou no motor de jogo",
}

INSTALL_CONJUNTO = '''#!/usr/bin/env bash
# Instala o que este pacote precisa em %(sistema)s. Uma venv so, usada por todas as pecas.
set -uo pipefail
cd "$(dirname "$0")"
FALHAS=""
SUDO=""
[ "$(id -u)" = 0 ] || { command -v sudo >/dev/null && SUDO="sudo"; }

%(basico)s
PY=${PY:-python3}
$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install %(pacotes)s || { echo "FALHOU a instalacao dos pacotes"; exit 1; }
%(ollama)s
echo
echo "== o que veio neste pacote"
%(linhas)s
echo
echo "Cada pasta tem o seu README com os detalhes."
'''

OLLAMA_SH = '''
if ! command -v ollama >/dev/null; then
  echo "== instalando o Ollama (o conhecimento de texto precisa do modelo base)"
  curl -fsSL https://ollama.com/install.sh | sh || echo "instale o Ollama a mao: ollama.com"
fi
'''


def _readme_conjunto(itens: list[dict], name: str, sistema: str) -> str:
    """Indice do pacote: o que tem dentro, em cima de que roda, e o que NAO da para fazer."""
    linhas = [f"# {name}", "",
              f"Pacote com {len(itens)} conhecimento(s) treinado(s), montado para "
              f"**{(sistemas.por_id(sistema) or {}).get('nome', sistema)}**.", "",
              "```bash", "./instalar.sh", "```", "",
              "## O que veio", "",
              "| Tipo | Nome | Roda em cima de | Como usar |", "|---|---|---|---|"]
    for item in itens:
        como = COMO_USAR.get(item["tipo"], "veja o README da pasta").format(id=item["id"])
        linhas.append(f"| {item['tipo']} | `{item['id']}` | {item.get('modelo') or '—'} | `{como}` |")
    textos = [i for i in itens if i["tipo"] == "texto"]
    linhas += ["", "## O que este pacote NAO faz", ""]
    if len(textos) > 1:
        linhas.append(
            f"**Os {len(textos)} conhecimentos de texto nao viram um so.** Cada adaptador "
            "responde bem o que treinou e atrapalha o resto, entao eles carregam um de cada "
            "vez. Para ter os dois juntos numa resposta so, treine os arquivos JUNTOS na "
            "pagina Treino e exporte o resultado — medi as duas formas, e so essa funciona.")
    else:
        linhas.append("Cada peca roda por si. Um LoRA de imagem nao entra no modelo de texto, "
                      "e um de som nao entra no de imagem: sao modelos base diferentes.")
    bases = sorted({i.get("modelo") or "" for i in itens if i.get("modelo")})
    if bases:
        linhas += ["", "## Modelos de origem", "",
                   "Cada adaptador so funciona no modelo em que foi treinado:", ""]
        linhas += [f"* `{b}`" for b in bases]
    if any(i["tipo"] == "som-estilo" for i in itens):
        linhas += ["", "> **Licenca do som:** o MusicGen e CC-BY-NC. O LoRA e seu, mas o audio "
                   "gerado por ele nao pode ser vendido."]
    linhas += ["", f"Montado em {_now()}."]
    return "\n".join(linhas) + "\n"


def start_varios(ids: list[str], name: str = "", include_base: bool = False,
                 archive: bool = True, sistema: str = "ubuntu") -> dict:
    """Junta varios conhecimentos consolidados num pacote so.

    Nao e fusao: cada peca vai inteira, na sua pasta, com o que precisa para rodar. Fundir
    adaptadores foi testado neste projeto e piora tudo — o README do pacote explica isso a
    quem receber, para ninguem tentar de novo.
    """
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um pacote sendo montado"}
    if train.job_status()["state"] == "running":
        return {"accepted": False, "reason": "treino em andamento; espere ou pare"}
    disponiveis = {i["id"]: i for i in prontos()}
    escolhidos = [disponiveis[i] for i in ids if i in disponiveis]
    if not escolhidos:
        return {"accepted": False, "reason": "escolha pelo menos um conhecimento consolidado"}
    faltando = [i for i in ids if i not in disponiveis]
    if faltando:
        return {"accepted": False,
                "reason": f"nao estao consolidados (ou nao tem pesos): {', '.join(faltando)}"}
    if not sistemas.por_id(sistema):
        return {"accepted": False, "reason": "sistema desconhecido"}
    name = re.sub(r"[^a-z0-9_-]+", "-", (name or "pacote-completo").lower()).strip("-")
    dest = _jobs_dir() / name
    if dest.exists():
        return {"accepted": False, "reason": f"ja existe um pacote chamado {name}"}
    rows = {r["id"]: r for r in train._rows() if r.get("snapshot")}
    _job.update({"state": "running", "lines": [], "name": name, "path": str(dest)})

    def _go() -> None:
        try:
            dest.mkdir(parents=True)
            pacotes: list[str] = []
            for item in escolhidos:
                sub = dest / item["tipo"] / item["id"]
                sub.mkdir(parents=True)
                _say(f"empacotando {item['tipo']}: {item['id']}")
                if item["tipo"] == "texto":
                    _empacota_texto(item["id"], rows[item["id"]], sub, item["id"], sistema,
                                    include_base)
                else:
                    _empacota_outro(item, sub, sistema)
                for pacote in PIP_POR_TIPO.get(item["tipo"], []):
                    if pacote not in pacotes:
                        pacotes.append(pacote)
            _say("montando o instalador do conjunto")
            linhas = "\n".join(
                f'echo "  {i["tipo"]}: {i["id"]}"' for i in escolhidos)
            script = INSTALL_CONJUNTO % {
                "sistema": (sistemas.por_id(sistema) or {}).get("nome", sistema),
                "basico": sistemas.bloco_basico(sistema),
                # sem aspas, o ">=" do pip viraria redirecionamento do shell
                "pacotes": " ".join(shlex.quote(x) for x in pacotes) or "pip",
                "ollama": OLLAMA_SH if any(i["tipo"] == "texto" for i in escolhidos) else "",
                "linhas": linhas,
            }
            alvo = dest / "instalar.sh"
            alvo.write_text(script, encoding="utf-8")
            alvo.chmod(0o755)
            (dest / "README.md").write_text(_readme_conjunto(escolhidos, name, sistema),
                                            encoding="utf-8")
            (dest / "pacote.json").write_text(json.dumps({
                "conjunto": True, "itens": [
                    {"tipo": i["tipo"], "id": i["id"], "modelo": i.get("modelo", ""),
                     "detalhe": i.get("detalhe", "")} for i in escolhidos],
                "sistema": sistema, "pip": pacotes, "criado": _now(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            _say(f"pacote montado ({_size(dest) / 1e6:.0f} MB)")
            if archive:
                _say("compactando para upload")
                tgz = _jobs_dir() / f"{name}.tar.gz"
                with tarfile.open(tgz, "w:gz") as tar:
                    tar.add(dest, arcname=name)
                _say(f"arquivo pronto: {tgz.name} ({tgz.stat().st_size / 1e6:.0f} MB)")
            _job["state"] = "done"
            _say("copie a pasta (ou o .tar.gz) para a outra maquina e rode ./instalar.sh")
        except Exception as exc:
            shutil.rmtree(dest, ignore_errors=True)
            _job["state"] = "error"
            _say(f"erro: {exc}")

    threading.Thread(target=_go, name="export-conjunto", daemon=True).start()
    return {"accepted": True, "name": name, "itens": len(escolhidos)}


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
