"""Diz, em uma olhada, o que esta ligado e o que falta instalar.

Ao abrir a pagina a ferramenta tenta falar com tudo: o Ollama (aqui ou em outro servidor),
o modelo escolhido, as dependencias de treino, a GPU e o conhecimento treinado. O que nao
responder aparece como "nao conectado", com o que fazer para resolver.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from agentepc import ollama, train
from agentepc.config import load


def _hf_baixado(repo: str) -> bool:
    """O modelo base do treino fica no cache do Hugging Face, nao no Ollama."""
    pasta = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    alvo = pasta / "hub" / ("models--" + repo.replace("/", "--"))
    return alvo.exists() and any(alvo.rglob("*.safetensors"))


def _gpu() -> dict:
    try:
        saida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if saida.returncode == 0 and saida.stdout.strip():
            nome, total, livre = [x.strip() for x in saida.stdout.strip().splitlines()[0].split(",")]
            return {"tem": True, "nome": nome, "vram_gb": round(int(total) / 1024, 1),
                    "livre_gb": round(int(livre) / 1024, 1)}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {"tem": False}


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for linha in fh:
                if linha.startswith("MemTotal:"):
                    return round(int(linha.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return 0.0


def _docker() -> dict:
    """Docker e opcional (serve para a pagina), mas o erro dele confunde: "command not found"
    quando o binario esta so na pasta do usuario, e "permission denied" quando o usuario
    nao esta no grupo docker. Aqui a diferenca aparece escrita."""
    caminhos = [shutil.which("docker"), str(Path.home() / ".local" / "bin" / "docker")]
    binario = next((c for c in caminhos if c and Path(c).exists()), "")
    if not binario:
        return {"estado": "aviso", "detalhe": "nao instalado",
                "acao": "opcional: serve para rodar a pagina em container"}
    achado = None
    sockets = [("sistema", "unix:///var/run/docker.sock"),
               ("do usuario", f"unix://{os.environ.get('XDG_RUNTIME_DIR', '/run/user/%d' % os.getuid())}/docker.sock")]
    negado = False
    for nome, host in sockets:
        env = {**os.environ, "DOCKER_HOST": host}
        try:
            r = subprocess.run([binario, "info", "--format", "{{.ServerVersion}}"],
                               env=env, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            achado = {"estado": "ok", "detalhe": f"{r.stdout.strip()} ({nome})", "acao": ""}
            continue
        if "permission denied" in (r.stdout + r.stderr).lower():
            negado = True
    if achado and negado:
        # dois Dockers na maquina: um funciona, o outro recusa. Dizer isso evita a confusao
        # de "o docker sumiu" quando na verdade e permissao no socket do outro.
        achado["detalhe"] += " — o Docker do sistema esta rodando mas nega acesso"
        achado["acao"] = "para usar o do sistema: sudo usermod -aG docker $USER e abra um terminal novo"
        return achado
    if achado:
        return achado
    if negado:
        return {"estado": "aviso", "detalhe": "rodando, mas sem permissao no socket",
                "acao": "sudo usermod -aG docker $USER e abra um terminal novo (wsl --shutdown)"}
    return {"estado": "aviso", "detalhe": "instalado, mas parado", "acao": "./docker/subir-docker.sh"}


def _treino_ok() -> dict:
    """Importa o que o treino precisa, no Python que roda a pagina."""
    faltando = []
    versoes = {}
    for mod in ("torch", "peft", "trl", "transformers", "datasets"):
        try:
            m = __import__(mod)
            versoes[mod] = getattr(m, "__version__", "?")
        except ImportError:
            faltando.append(mod)
    cuda = False
    if "torch" in versoes:
        import torch

        cuda = bool(torch.cuda.is_available())
    return {"faltando": faltando, "versoes": versoes, "cuda": cuda}


def catalogo() -> list[dict]:
    cfg = load()
    ativo = (cfg.get("model") or {}).get("name")
    gpu, ram = _gpu(), _ram_gb()
    try:
        instalados = set(ollama.tags())
    except Exception:
        instalados = set()
    saida = []
    for m in cfg.get("modelos") or []:
        # VRAM manda no caminho da GPU; RAM manda no caminho da memoria. Sao coisas separadas:
        # o 3B treina numa placa de 8 GB mesmo com 15 GB de RAM.
        vram = float(m.get("vram_gb") or 0)
        ram_cpu = float(m.get("ram_gb") or 0)
        folga = 1.08  # MemTotal sempre aparece um pouco abaixo do nominal
        cabe_gpu = gpu["tem"] and vram and gpu.get("vram_gb", 0) * folga >= vram
        cabe_ram = ram * folga >= ram_cpu
        if cabe_gpu:
            roda = "sim, na GPU"
        elif m.get("sem_gpu") and cabe_ram:
            roda = "sim, na memória (mais lento)"
        elif ram * folga >= 8:
            roda = "só conversa; treinar exige mais VRAM" if gpu["tem"] else "só conversa; treinar exige GPU"
        else:
            roda = "não cabe nesta máquina"
        saida.append({
            **m,
            "ativo": m.get("name") == ativo,
            "baixado_ollama": m.get("name") in instalados,
            "baixado_treino": _hf_baixado(m.get("base_hf", "")),
            "roda_aqui": roda,
        })
    return saida


def checar() -> dict:
    """Um item por coisa que precisa estar de pe. estado: ok, falta ou aviso."""
    cfg = load()
    mcfg = cfg.get("model") or {}
    itens = []

    host = ollama.host()
    remoto = not ollama._is_local_host(host)
    ligado = ollama.ping()
    itens.append({
        "chave": "ollama",
        "titulo": f"Ollama ({'servidor ' + host if remoto else 'local'})",
        "estado": "ok" if ligado else "falta",
        "detalhe": "conectado" if ligado else "não conectado",
        "acao": "" if ligado else (
            f"suba o Ollama em {host}" if remoto else "use Ambiente > Ollama, ou rode: ollama serve"
        ),
    })

    tags = ollama.tags() if ligado else []
    tem_modelo = any(t == mcfg.get("name") or t.startswith(str(mcfg.get("name")).split(":")[0] + ":") for t in tags)
    itens.append({
        "chave": "modelo",
        "titulo": f"Modelo do chat ({mcfg.get('name')})",
        "estado": "ok" if tem_modelo else "falta",
        "detalhe": "baixado" if tem_modelo else ("não baixado" if ligado else "sem Ollama, não dá para saber"),
        "acao": "" if tem_modelo else f"ollama pull {mcfg.get('name')}",
    })

    gpu = _gpu()
    ram = _ram_gb()
    escolha = (mcfg.get("device") or "auto").lower()
    itens.append({
        "chave": "gpu",
        "titulo": "Placa de vídeo",
        "estado": "ok" if gpu["tem"] else "aviso",
        "detalhe": (f"{gpu['nome']}, {gpu['vram_gb']} GB ({gpu['livre_gb']} GB livres)" if gpu["tem"]
                    else f"nenhuma; {ram} GB de RAM"),
        "acao": "" if gpu["tem"] else "sem GPU, escolha um modelo marcado 'treina na memória'",
    })

    treino = _treino_ok()
    itens.append({
        "chave": "treino",
        "titulo": "Dependências de treino",
        "estado": "ok" if not treino["faltando"] else "falta",
        "detalhe": (", ".join(f"{k} {v}" for k, v in treino["versoes"].items()) or "nada instalado")
        + (f" — faltando: {', '.join(treino['faltando'])}" if treino["faltando"] else "")
        + (" — CUDA ativa" if treino["cuda"] else " — sem CUDA (treino na memória)"),
        "acao": "" if not treino["faltando"] else "Ambiente > ambiente de treino",
    })

    base_ok = _hf_baixado(mcfg.get("base_hf", ""))
    itens.append({
        "chave": "base_treino",
        "titulo": f"Pesos para treinar ({mcfg.get('base_hf')})",
        "estado": "ok" if base_ok else "aviso",
        "detalhe": "no cache" if base_ok else "baixa sozinho no primeiro treino (alguns GB)",
        "acao": "",
    })

    adaptador = train.serving_adapter()
    consolidados = train.consolidated_files()
    cat = train.adapters()
    incompativeis = [i for i in cat["items"] if not i.get("serve_no_modelo")]
    itens.append({
        "chave": "lora",
        "titulo": "Conhecimento treinado (LoRA)",
        "estado": "ok" if adaptador else "aviso",
        "detalhe": (f"ativo: {adaptador.name} — {len(consolidados)} consolidado(s)" if adaptador
                    else "nenhum treinado ainda")
        + (f" — {len(incompativeis)} treinado(s) em outro modelo, não servem no atual"
           if incompativeis else ""),
        "acao": ("" if adaptador else "treine um arquivo na página Treino")
        or ("treine de novo no modelo atual para usá-los" if incompativeis else ""),
    })

    dk = _docker()
    itens.append({"chave": "docker", "titulo": "Docker (opcional)", **dk})

    livre = round(shutil.disk_usage(".").free / 1e9, 1)
    itens.append({
        "chave": "disco",
        "titulo": "Espaço em disco",
        "estado": "ok" if livre > 20 else "aviso",
        "detalhe": f"{livre} GB livres",
        "acao": "" if livre > 20 else "cada modelo base ocupa alguns GB",
    })

    faltas = [i for i in itens if i["estado"] == "falta"]
    return {
        "conectado": not faltas,
        "resumo": ("tudo pronto" if not faltas else "falta: " + ", ".join(i["titulo"] for i in faltas)),
        "itens": itens,
        "gpu": gpu,
        "ram_gb": ram,
        "device": escolha,
        "ollama_host": host,
        "remoto": remoto,
        "modelos": catalogo(),
    }


def hub_dir() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _orfaos() -> list[Path]:
    """Blobs que nenhum snapshot aponta mais.

    O cache do Hugging Face guarda o peso em hub/blobs e cada modelo so tem um link
    para ele. Apagar a pasta do modelo nao libera espaco nenhum — foi o que aconteceu
    aqui: 6,5 GB ficaram para tras.
    """
    hub = hub_dir()
    if not hub.exists():
        return []
    usados = set()
    for link in hub.rglob("snapshots/*/*"):
        if link.is_symlink():
            try:
                usados.add(link.resolve())
            except OSError:
                pass
    # o conteudo fica em blobs/<2 letras>/<sha256>, nao solto em blobs/
    sobra = []
    for blob in (hub / "blobs").rglob("*"):
        if not blob.is_file() or blob.suffix in (".lock", ".refs"):
            continue
        if blob.resolve() not in usados:
            sobra.append(blob)
    return sobra


def apagar_pesos(repo: str) -> int:
    """Tira do cache os pesos daquele modelo e os blobs que ficarem sem dono."""
    import shutil

    pasta = hub_dir() / ("models--" + repo.replace("/", "--"))
    if pasta.exists():
        shutil.rmtree(pasta, ignore_errors=True)
    return _remove_orfaos()


def _remove_orfaos() -> int:
    livre = 0
    for blob in _orfaos():
        try:
            livre += blob.stat().st_size
            blob.unlink()
            for extra in (blob.with_suffix(blob.suffix + ".lock"), Path(str(blob) + ".refs")):
                extra.unlink(missing_ok=True)
        except OSError:
            pass
    return livre


def limpar_cache() -> dict:
    """Varre blobs orfaos (de trocas antigas) e devolve quanto liberou."""
    return {"ok": True, "liberado_gb": round(_remove_orfaos() / 1e9, 2)}


_troca: dict = {"state": "idle", "lines": [], "alvo": ""}


def troca_status() -> dict:
    return dict(_troca)


def _dizer(msg: str) -> None:
    _troca["lines"].append(msg)
    del _troca["lines"][:-80]


def plano_troca(model_id: str) -> dict:
    """O que sai e o que entra, para você confirmar antes."""
    cfg = load()
    novo = next((m for m in cfg.get("modelos") or [] if m.get("id") == model_id), None)
    if not novo:
        raise ValueError("modelo desconhecido")
    atual = cfg.get("model") or {}
    treinos = [r for r in train._rows() if r.get("snapshot")]
    perdidos = [r for r in treinos if (r.get("base_hf") or atual.get("base_hf")) != novo["base_hf"]]
    return {
        "novo": novo,
        "atual": {"name": atual.get("name"), "base_hf": atual.get("base_hf")},
        "remove_ollama": atual.get("name") if atual.get("name") != novo["name"] else "",
        "remove_pesos": atual.get("base_hf") if atual.get("base_hf") != novo["base_hf"] else "",
        "perde_conhecimento": [{"id": r["id"], "file": r.get("file")} for r in perdidos],
    }


def trocar_modelo(model_id: str, apagar_conhecimento: bool = True, baixar_pesos: bool = True) -> dict:
    """Desinstala o modelo antigo (Ollama + pesos de treino) e instala o escolhido."""
    if _troca["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma troca rodando"}
    plano = plano_troca(model_id)
    _troca.update({"state": "running", "lines": [], "alvo": plano["novo"]["name"]})

    def _go() -> None:
        import shutil
        import subprocess

        try:
            from agentepc import engine

            # 1. o conhecimento treinado no modelo antigo nao serve no novo
            if apagar_conhecimento and plano["perde_conhecimento"]:
                arquivos = {p["file"] for p in plano["perde_conhecimento"]}
                for arq in arquivos:
                    train.delete_group(arq)
                _dizer(f"apaguei {len(plano['perde_conhecimento'])} conhecimento(s) do modelo antigo")
            engine.unload()
            try:
                ollama.stop_model()
            except Exception:
                pass

            # 2. tira o modelo antigo do Ollama
            if plano["remove_ollama"]:
                _dizer(f"removendo {plano['remove_ollama']} do Ollama")
                subprocess.run([ollama.ollama_bin(), "rm", plano["remove_ollama"]],
                               env=ollama._env(), check=False, capture_output=True, timeout=120)

            # 3. e os pesos de treino dele do cache
            if plano["remove_pesos"]:
                livre = apagar_pesos(plano["remove_pesos"])
                _dizer(f"apaguei os pesos de treino de {plano['remove_pesos']} ({livre / 1e9:.1f} GB liberados)")

            # 4. grava a escolha e instala o novo
            usar_modelo(model_id)
            _dizer(f"modelo agora e {plano['novo']['name']}")
            _dizer(f"baixando {plano['novo']['name']} no Ollama")
            proc = subprocess.run([ollama.ollama_bin(), "pull", plano["novo"]["name"]],
                                  env=ollama._env(), capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                raise RuntimeError(f"ollama pull falhou: {(proc.stderr or '')[-200:]}")
            _dizer("modelo do chat pronto")

            if baixar_pesos:
                try:
                    from huggingface_hub import snapshot_download

                    _dizer(f"baixando os pesos de treino ({plano['novo']['base_hf']}) — pode demorar")
                    snapshot_download(
                        plano["novo"]["base_hf"],
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                    )
                    _dizer("pesos de treino no cache")
                except ImportError:
                    _dizer("sem huggingface_hub aqui: os pesos de treino baixam sozinhos no 1o treino")
                except Exception as exc:
                    _dizer(f"nao baixei os pesos de treino agora ({exc}); baixam no 1o treino")
            _troca["state"] = "done"
            _dizer("pronto")
        except Exception as exc:
            _troca["state"] = "error"
            _dizer(f"erro: {exc}")

    import threading

    threading.Thread(target=_go, name="troca-modelo", daemon=True).start()
    return {"accepted": True, **plano}


def usar_modelo(model_id: str) -> dict:
    """So troca a escolha no config (sem mexer no que esta instalado)."""
    import re

    cfg = load()
    alvo = next((m for m in cfg.get("modelos") or [] if m.get("id") == model_id), None)
    if not alvo:
        raise ValueError("modelo desconhecido")
    from agentepc.config import CONFIG_PATH

    texto = CONFIG_PATH.read_text(encoding="utf-8")
    cabeca, resto = texto.split("\nmodelos:", 1)
    cabeca = re.sub(r"(?m)^(\s*name:\s*).*$", lambda m: m.group(1) + alvo["name"], cabeca, count=1)
    cabeca = re.sub(r"(?m)^(\s*base_hf:\s*).*$", lambda m: m.group(1) + alvo["base_hf"], cabeca, count=1)
    CONFIG_PATH.write_text(cabeca + "\nmodelos:" + resto, encoding="utf-8")
    try:
        from agentepc import engine

        engine.unload()
    except Exception:
        pass
    return {"ok": True, "modelo": alvo["name"], "base_hf": alvo["base_hf"]}
