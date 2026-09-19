"""Prepara uma maquina para rodar e treinar o modelo — esta ou outra, por SSH.

O alvo pode ser o mesmo servidor da sua API ou uma maquina separada so para o treino.
O que sai daqui e um script: ele e montado conforme o sistema detectado, mostrado antes
de rodar e executado com a saida ao vivo. Nada roda sem voce mandar.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from agentepc import sistemas
from agentepc.config import ROOT, load

# Um unico comando que descreve a maquina. Serve em qualquer shell POSIX.
SNIFF_POSIX = r"""
echo "os=$(uname -s)"
echo "arch=$(uname -m)"
. /etc/os-release 2>/dev/null && echo "distro=$ID" && echo "versao=$VERSION_ID" || echo "distro=?"
for p in apt-get dnf yum zypper pacman apk brew; do command -v $p >/dev/null && echo "pkg=$p" && break; done
echo "root=$( [ "$(id -u)" = 0 ] && echo sim || (sudo -n true 2>/dev/null && echo sudo || echo nao) )"
echo "python=$(python3 -V 2>&1 | cut -d' ' -f2)"
echo "cc=$(command -v cc gcc 2>/dev/null | head -1)"
echo "docker=$(command -v docker >/dev/null && docker --version 2>/dev/null | cut -d' ' -f3 | tr -d , || echo nao)"
echo "ollama=$(command -v ollama >/dev/null || [ -x "$HOME/.local/bin/ollama" ] && echo sim || echo nao)"
echo "gpu=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo nenhuma)"
echo "ram=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo ?)"
"""

SNIFF_WIN = r"""
Write-Output "os=Windows"
Write-Output "arch=$env:PROCESSOR_ARCHITECTURE"
Write-Output "versao=$([System.Environment]::OSVersion.Version)"
Write-Output "pkg=$(if (Get-Command winget -EA 0) {'winget'} elseif (Get-Command choco -EA 0) {'choco'} else {'?'})"
Write-Output "root=$(if (([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole('Administrators')) {'sim'} else {'nao'})"
Write-Output "python=$(if (Get-Command python -EA 0) {(python -V) -replace 'Python ',''} else {'nao'})"
Write-Output "docker=$(if (Get-Command docker -EA 0) {(docker --version)} else {'nao'})"
Write-Output "ollama=$(if (Get-Command ollama -EA 0) {'sim'} else {'nao'})"
Write-Output "gpu=$(if (Get-Command nvidia-smi -EA 0) {(nvidia-smi --query-gpu=name --format=csv,noheader)} else {'nenhuma'})"
"""

# ---------------------------------------------------------------- receitas
BASE_LINUX = r"""#!/usr/bin/env bash
# Preparado por agente-pc para {distro} ({pkg}). Roda de novo sem estragar nada.
# sem -e: uma etapa que falha nao pode impedir as outras; o fim mostra o que falhou
set -uo pipefail
FALHAS=""
SENHA='{senha}'
# a senha chega pelo stdin do bash, nunca em arquivo nem na lista de processos
sudo_() {{
  if [ -n "$SENHA" ]; then printf '%s\n' "$SENHA" | sudo -S -p '' "$@"
  else sudo -n "$@"; fi
}}
SUDO=""
if [ "$(id -u)" != 0 ] && command -v sudo >/dev/null; then SUDO="sudo_"; fi
diga() {{ printf '\n== %s\n' "$1"; }}

instalar() {{
  case "{pkg}" in
    apt-get) $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y "$@" ;;
    dnf)     $SUDO dnf install -y "$@" ;;
    yum)     $SUDO yum install -y "$@" ;;
    zypper)  $SUDO zypper --non-interactive install "$@" ;;
    pacman)  $SUDO pacman -Sy --noconfirm "$@" ;;
    apk)     $SUDO apk add --no-cache "$@" ;;
    brew)    brew install "$@" ;;
    *) echo "gerenciador de pacotes desconhecido; instale a mao: $*"; return 1 ;;
  esac
}}
"""

PASSO_BASICO = r"""
{basico}"""

PASSO_DOCKER = r"""
diga "docker"
if command -v docker >/dev/null; then
  echo "ja instalado: $(docker --version)"
else
  curl -fsSL https://get.docker.com | $SUDO sh
  $SUDO usermod -aG docker "$(id -un)" 2>/dev/null || true
  echo "entre e saia da sessao para usar docker sem sudo"
fi
"""

PASSO_OLLAMA = r"""
diga "ollama"
ok_ollama() {{
  command -v ollama >/dev/null || [ -x "$HOME/.local/bin/ollama" ]
}}
if command -v ollama >/dev/null || [ -x "$HOME/.local/bin/ollama" ]; then
  echo "ja instalado"
elif [ -n "$SUDO" ] || [ "$(id -u)" = 0 ]; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  # sem root: vai para a pasta do usuario, do mesmo jeito que foi feito na maquina de origem
  mkdir -p "$HOME/.local"
  curl -fL --retry 3 -o /tmp/ollama.tgz https://ollama.com/download/ollama-linux-amd64.tgz \
    || curl -fL --retry 3 -o /tmp/ollama.tgz https://ollama.com/download/ollama-linux-amd64.tar.zst
  (tar -xzf /tmp/ollama.tgz -C "$HOME/.local" 2>/dev/null || tar --zstd -xf /tmp/ollama.tgz -C "$HOME/.local")
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
fi
export PATH="$HOME/.local/bin:$PATH"
if ok_ollama; then
  (pgrep -f "ollama serve" >/dev/null) || (nohup ollama serve >/tmp/ollama-serve.log 2>&1 & sleep 3)
  echo "OK ollama"
else
  echo "FALHOU ollama — nao consegui instalar"
  FALHAS="$FALHAS ollama"
fi
"""

PASSO_MODELO = r"""
diga "baixando o modelo {modelo}"
export PATH="$HOME/.local/bin:$PATH"
if ollama pull {modelo}; then echo "OK modelo"; else
  echo "FALHOU modelo — o Ollama respondeu?"
  FALHAS="$FALHAS modelo"
fi
"""

PASSO_TREINO = r"""
diga "ambiente de treino (torch, peft, trl)"
cd "{destino}"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-train 2>/dev/null || python3 -m venv .venv-train
if command -v uv >/dev/null; then
  uv pip install --python .venv-train/bin/python -r requirements-train.txt
else
  .venv-train/bin/pip install -r requirements-train.txt
fi
if .venv-train/bin/python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "OK treino"
else
  echo "FALHOU treino — veja o erro do pip acima"
  FALHAS="$FALHAS treino"
fi
"""

PASSO_CAPACIDADES = r"""
diga "treinos escolhidos: {nomes}"
cd "{destino}"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x .venv-train/bin/python ] || uv venv .venv-train 2>/dev/null || python3 -m venv .venv-train
if command -v uv >/dev/null; then
  uv pip install --python .venv-train/bin/python {pacotes}
else
  .venv-train/bin/pip install {pacotes}
fi
if [ $? -eq 0 ]; then echo "OK treinos"; else
  echo "FALHOU treinos — veja o erro do pip acima"
  FALHAS="$FALHAS treinos"
fi
"""

FECHO = r"""
if [ -z "$FALHAS" ]; then echo "TUDO-OK"; else echo "ETAPAS-COM-FALHA:$FALHAS"; fi
"""

PASSO_PAGINA = r"""
diga "subindo a pagina"
cd "{destino}"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv-train/bin/python
[ -x "$PY" ] || PY=python3
PYTHONPATH=src nohup $PY -m agentepc web --host 0.0.0.0 --port {porta} >/tmp/agentepc-web.log 2>&1 &
sleep 3
echo "pagina em http://SEU-IP:{porta}"
"""

BASE_WIN = r"""# Preparado por agente-pc para Windows. Roda de novo sem estragar nada.
$ErrorActionPreference = "Stop"
function Diga($m) {{ Write-Output "`n== $m" }}
function Instalar($id) {{
  if (Get-Command winget -EA 0) {{ winget install --silent --accept-package-agreements --accept-source-agreements --id $id }}
  elseif (Get-Command choco -EA 0) {{ choco install -y $id }}
  else {{ Write-Output "instale a mao: $id" }}
}}
"""

WIN_BASICO = r"""
Diga "basico: python e git"
if (-not (Get-Command python -EA 0)) {{ Instalar Python.Python.3.12 }}
if (-not (Get-Command git -EA 0)) {{ Instalar Git.Git }}
"""
WIN_DOCKER = r"""
Diga "docker"
if (Get-Command docker -EA 0) {{ docker --version }} else {{ Instalar Docker.DockerDesktop }}
"""
WIN_OLLAMA = r"""
Diga "ollama"
if (Get-Command ollama -EA 0) {{ Write-Output "ja instalado" }} else {{ Instalar Ollama.Ollama }}
"""
WIN_MODELO = r"""
Diga "baixando o modelo {modelo}"
ollama pull {modelo}
"""
WIN_CAPACIDADES = r"""
Diga "treinos escolhidos: {nomes}"
Set-Location "{destino}"
if (-not (Test-Path .\.venv-train)) {{ python -m venv .venv-train }}
.\.venv-train\Scripts\python.exe -m pip install {pacotes}
"""

WIN_TREINO = r"""
Diga "ambiente de treino"
Set-Location "{destino}"
python -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install --upgrade pip
.\.venv-train\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv-train\Scripts\python.exe -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"""
WIN_PAGINA = r"""
Diga "subindo a pagina"
Set-Location "{destino}"
$env:PYTHONPATH = "src"
Start-Process -WindowStyle Hidden .\.venv-train\Scripts\python.exe -ArgumentList "-m","agentepc","web","--host","0.0.0.0","--port","{porta}"
Write-Output "pagina em http://SEU-IP:{porta}"
"""


_job: dict = {"state": "idle", "lines": [], "alvo": "", "script": "", "info": {}, "sistema": "", "falhas": []}


def _say(msg: str) -> None:
    _job["lines"].append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "msg": msg})
    del _job["lines"][:-400]


def status() -> dict:
    return dict(_job)


# ------------------------------------------------------------------ alvo
def ssh_args(destino: str, porta: int = 22, chave: str = "") -> list[str]:
    if not re.fullmatch(r"[\w.-]+@[\w.-]+", destino or ""):
        raise ValueError("use usuario@servidor")
    args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15"]
    if porta and int(porta) != 22:
        args += ["-p", str(int(porta))]
    if chave:
        args += ["-i", str(Path(chave).expanduser())]
    return args + [destino]


def rodar(script: str, alvo: dict, windows: bool, ao_vivo=None, timeout: int = 3600) -> int:
    """Executa o script aqui ou no servidor remoto, jogando a saida linha a linha."""
    if alvo["tipo"] == "ssh":
        shell = ["powershell", "-NoProfile", "-Command", "-"] if windows else ["bash", "-s"]
        cmd = ssh_args(alvo["destino"], alvo.get("porta", 22), alvo.get("chave", "")) + shell
    else:
        cmd = ["powershell.exe", "-NoProfile", "-Command", "-"] if windows else ["bash", "-s"]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(script)
    proc.stdin.close()
    for linha in proc.stdout:
        linha = linha.rstrip()
        if linha and ao_vivo:
            ao_vivo(linha)
    return proc.wait(timeout=timeout)


def detectar(alvo: dict) -> dict:
    """Le a maquina alvo. Tenta POSIX; se falhar e for SSH, tenta PowerShell."""
    achado: dict = {}

    def coleta(linha: str) -> None:
        if "=" in linha:
            k, v = linha.split("=", 1)
            achado[k.strip()] = v.strip()

    try:
        code = rodar(SNIFF_POSIX, alvo, windows=False, ao_vivo=coleta, timeout=120)
    except Exception as exc:
        code, achado = 1, {"erro": str(exc)}
    if code != 0 or not achado.get("os"):
        try:
            achado = {}
            rodar(SNIFF_WIN, alvo, windows=True, ao_vivo=coleta, timeout=120)
        except Exception as exc:
            achado.setdefault("erro", str(exc))
    achado["windows"] = achado.get("os") == "Windows"
    return achado


MARCA_SENHA = "__SENHA__"


def caminho_shell(caminho: str) -> str:
    """Prepara a pasta de destino para entrar no script.

    Dentro de aspas o `~` NAO expande: `cd "~/agente-pc"` procura uma pasta chamada `~`.
    Trocando por `$HOME`, que expande entre aspas, o mesmo caminho funciona. Tambem corto
    caractere que nao deveria estar num caminho — ele vai parar dentro de um comando.
    """
    import re as _re

    caminho = (caminho or "").strip()
    caminho = _re.sub(r"[^\w./~-]", "", caminho)
    if caminho.startswith("~/"):
        return "$HOME/" + caminho[2:]
    if caminho == "~":
        return "$HOME"
    return caminho or "$HOME/agente-pc"


def montar_script(info: dict, opcoes: dict, senha: str = MARCA_SENHA) -> str:
    from agentepc import capacidades

    modelo = (load().get("model") or {}).get("name") or "qwen2.5:3b"
    treinos = [t for t in (opcoes.get("treinos") or []) if capacidades.PACOTES.get(t)]
    pacotes = " ".join(shlex.quote(p) for p in capacidades.pacotes_de(treinos))
    nomes_treinos = capacidades.nomes_de(treinos)
    destino = caminho_shell(opcoes.get("destino_dir") or str(ROOT))
    porta = int(opcoes.get("porta") or 8765)
    if info.get("windows"):
        partes = [BASE_WIN]
        if opcoes.get("basico", True):
            partes.append(WIN_BASICO)
        if opcoes.get("ollama", True):
            partes.append(WIN_OLLAMA)
        if opcoes.get("modelo", True):
            partes.append(WIN_MODELO)
        if opcoes.get("treino"):
            partes.append(WIN_TREINO)
        if treinos:
            partes.append(WIN_CAPACIDADES)
        return "".join(p.format(modelo=modelo, destino=destino, porta=porta, basico="", senha="",
                                pacotes=pacotes, nomes=nomes_treinos) for p in partes)

    pkg = info.get("pkg", "apt-get")
    sistema = opcoes.get("sistema") or sistemas.detecta(info)
    partes = [BASE_LINUX]
    if opcoes.get("basico", True):
        partes.append(PASSO_BASICO)
    if opcoes.get("ollama", True):
        partes.append(PASSO_OLLAMA)
    if opcoes.get("modelo", True):
        partes.append(PASSO_MODELO)
    if opcoes.get("treino"):
        partes.append(PASSO_TREINO)
    if treinos:
        partes.append(PASSO_CAPACIDADES)
    partes.append(FECHO)
    basico = sistemas.bloco_basico(sistema)
    return "".join(
        p.format(pkg=pkg, distro=info.get("distro", "?"), modelo=modelo, destino=destino,
                 porta=porta, basico=basico, senha=senha, pacotes=pacotes, nomes=nomes_treinos)
        for p in partes
    )


def enviar_projeto(alvo: dict, destino_dir: str) -> None:
    """Manda o codigo (sem data/ nem venv) para o servidor, por tar sobre SSH."""
    tar = subprocess.Popen(
        ["tar", "czf", "-", "--exclude=./data", "--exclude=./.venv-train", "--exclude=./.git",
         "--exclude=./data/adapters", "-C", str(ROOT), "."],
        stdout=subprocess.PIPE,
    )
    pasta = caminho_shell(destino_dir)
    alvo_cmd = ssh_args(alvo["destino"], alvo.get("porta", 22), alvo.get("chave", "")) + [
        f'mkdir -p "{pasta}" && tar xzf - -C "{pasta}"'
    ]
    subprocess.run(alvo_cmd, stdin=tar.stdout, check=True, timeout=600)
    tar.wait()


def preparar(opcoes: dict, executar: bool) -> dict:
    """Detecta a maquina e monta o script. Com executar=True, tambem roda."""
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma preparacao rodando"}
    tipo = "ssh" if (opcoes.get("destino") or "").strip() else "local"
    alvo = {
        "tipo": tipo,
        "destino": (opcoes.get("destino") or "").strip(),
        "porta": int(opcoes.get("porta_ssh") or 22),
        "chave": (opcoes.get("chave") or "").strip(),
    }
    if tipo == "ssh":
        try:
            ssh_args(alvo["destino"], alvo["porta"], alvo["chave"])
        except ValueError as exc:
            return {"accepted": False, "reason": str(exc)}
    _job.update({"state": "running", "lines": [], "alvo": alvo["destino"] or "esta maquina",
                 "script": "", "info": {}, "sistema": opcoes.get("sistema") or "", "falhas": []})

    def _go() -> None:
        try:
            _say(f"olhando {'o servidor ' + alvo['destino'] if tipo == 'ssh' else 'esta maquina'}")
            info = detectar(alvo)
            _job["info"] = info
            if info.get("erro") and not info.get("os"):
                raise RuntimeError(
                    f"nao consegui falar com o alvo: {info['erro']}. "
                    "Por SSH e preciso chave configurada (sem senha interativa)."
                )
            sist = opcoes.get("sistema") or sistemas.detecta(info)
            _job["sistema"] = sist
            nome = (sistemas.por_id(sist) or {}).get("nome", sist)
            _say(f"sistema reconhecido: {nome}")
            _say("detalhes: " + json.dumps({k: v for k, v in info.items() if k != "erro"}, ensure_ascii=False))
            senha = (opcoes.get("senha") or "")
            precisa_senha = (
                not info.get("windows")
                and info.get("root") == "nao"
                and bool(opcoes.get("basico", True))
            )
            if precisa_senha and not senha:
                raise RuntimeError(
                    "esta maquina pede senha de administrador para instalar pacote do sistema. "
                    "Preencha a senha (local ou do servidor) e tente de novo — ou desmarque o basico."
                )
            if not info.get("windows") and info.get("root") == "nao":
                _say("sem sudo liberado: vou usar a senha informada para as etapas de sistema"
                     if senha else "sem root: o Ollama vai para a pasta do usuario")
            if info.get("gpu") in ("nenhuma", "", None):
                _say("sem GPU detectada: da para conversar, mas treinar fica inviavel")
            # o preview nunca mostra a senha; ela so entra na hora de executar
            _job["script"] = montar_script(info, opcoes, senha="***" if senha else "")
            script = montar_script(info, opcoes, senha=senha)
            if not executar:
                _job["state"] = "pronto"
                _say("script montado. confira acima e clique em Executar.")
                return
            if tipo == "ssh" and opcoes.get("enviar_projeto"):
                _say("enviando o projeto para o servidor")
                enviar_projeto(alvo, opcoes.get("destino_dir") or "~/agente-pc")
            _say("executando")
            falhas: list[str] = []

            def olho(linha: str) -> None:
                _say(linha)
                if linha.startswith("ETAPAS-COM-FALHA:"):
                    falhas.extend(linha.split(":", 1)[1].split())

            code = rodar(script, alvo, windows=bool(info.get("windows")), ao_vivo=olho)
            if falhas:
                _job["state"] = "aviso"
                _job["falhas"] = falhas
                _say(f"terminou, mas {len(falhas)} etapa(s) falharam: {', '.join(falhas)}")
            elif code == 0:
                _job["state"] = "done"
                _say("pronto")
            else:
                _job["state"] = "error"
                _say(f"terminou com erro (codigo {code})")
        except Exception as exc:
            _job["state"] = "error"
            _say(f"erro: {exc}")

    threading.Thread(target=_go, name="provision", daemon=True).start()
    return {"accepted": True}


def apontar_ollama(host: str) -> dict:
    """Passa o chat a usar o Ollama de outro servidor (o treino continua sendo local)."""
    host = (host or "").strip()
    cfg = ROOT / "config.yaml"
    texto = cfg.read_text(encoding="utf-8")
    novo = re.sub(r"(?m)^(\s*ollama_host:\s*).*$", lambda m: m.group(1) + (host or "127.0.0.1:11434"), texto)
    cfg.write_text(novo, encoding="utf-8")
    return {"ok": True, "ollama_host": host or "127.0.0.1:11434"}
