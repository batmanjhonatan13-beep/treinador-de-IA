"""Catalogo de treinos: o que existe, o que a maquina aguenta, o que falta instalar.

A ferramenta roda em maquinas muito diferentes. Em vez de oferecer tudo e quebrar na cara
de quem nao tem GPU, cada treino declara o que precisa e e conferido antes de aparecer:
VRAM, RAM, pacote Python, programa externo. O que nao passa fica visivel, mas bloqueado,
com o motivo escrito e o comando que resolve.

Ha tambem o que nao foi implementado. Esta listado do mesmo jeito, dizendo por que — e
melhor uma linha honesta do que um botao que promete e falha.
"""

from __future__ import annotations

import importlib.util
import shutil

from agentepc.health import _gpu, _ram_gb

# estado: "pronto" (implementado e testado) ou "planejado" (ainda nao existe)
CATALOGO: list[dict] = [
    {
        "id": "texto-fatos",
        "nome": "Fatos em texto (LoRA)",
        "faz": "Ensina informação a um modelo de linguagem e prova que ele aprendeu.",
        "pagina": "/treino.html",
        "estado": "pronto",
        "prova": "sim — 3 tentativas seguidas acertando a prova inteira sem consulta",
        "vram_gb": 0,          # roda na memória com modelo pequeno
        "ram_gb": 5,
        "modulos": ["torch", "peft", "trl", "transformers"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-train.txt",
        "nota": "Sem GPU funciona com SmolLM2/Qwen pequenos, mais devagar.",
    },
    {
        "id": "imagem-estilo",
        "nome": "Estilo de imagem (LoRA, SD 1.5)",
        "faz": "Ensina um traço ou estética ao Stable Diffusion.",
        "pagina": "/imagens.html",
        "estado": "pronto",
        "prova": "não — estilo se julga olhando as amostras",
        "vram_gb": 6,
        "ram_gb": 8,
        "modulos": ["torch", "diffusers", "peft", "torchvision"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-imagem.txt",
        "nota": "Difusão na CPU é inviável, não apenas lenta.",
    },
    {
        "id": "imagem-personagem",
        "nome": "Personagem ou objeto (LoRA com gatilho)",
        "faz": "Fixa uma pessoa, mascote ou produto para reaparecer igual em cada imagem.",
        "pagina": "/imagens.html",
        "estado": "pronto",
        "prova": "parcial — as amostras mostram se a identidade se manteve",
        "vram_gb": 6,
        "ram_gb": 8,
        "modulos": ["torch", "diffusers", "peft", "torchvision"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-imagem.txt",
        "nota": "Peça 10–25 fotos do mesmo assunto, ângulos variados, fundo diferente.",
    },
    {
        "id": "imagem-sdxl",
        "nome": "Estilo em SDXL (mais qualidade)",
        "faz": "O mesmo treino de estilo, num modelo maior e mais detalhado.",
        "pagina": "/imagens.html",
        "estado": "pronto",
        "prova": "não — julgue pelas amostras",
        "vram_gb": 12,
        "ram_gb": 16,
        "modulos": ["torch", "diffusers", "peft", "torchvision"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-imagem.txt",
        "nota": "Numa placa de 12 GB fica apertado: use 512 px e passos curtos.",
    },
    {
        "id": "classificador-imagem",
        "nome": "Classificador de imagem",
        "faz": "Aprende a separar categorias suas (peça boa × com defeito, por exemplo).",
        "pagina": "/classificador.html",
        "estado": "pronto",
        "prova": "sim — acerto medido num conjunto que ele nunca viu",
        "vram_gb": 0,
        "ram_gb": 6,
        "modulos": ["torch", "torchvision"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-imagem.txt",
        "nota": "É o único além do texto com prova automática de verdade.",
    },
    {
        "id": "classificador-som",
        "nome": "Classificador de som",
        "faz": "Separa sons em categorias suas (motor batendo × rodando liso, por exemplo).",
        "pagina": "/som.html",
        "estado": "pronto",
        "prova": "sim — acerto medido em faixas que ele nunca ouviu",
        "vram_gb": 0,
        "ram_gb": 6,
        "modulos": ["torch", "torchvision", "soundfile", "scipy"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-som.txt",
        "nota": "O áudio vira espectrograma — uma imagem do som — e daí é o mesmo treino "
                "do classificador de imagem. Roda sem placa de vídeo.",
    },
    {
        "id": "som-gerar",
        "nome": "Gerar som e música (MusicGen)",
        "faz": "Você escreve \"chuva com trovão ao longe\" e sai um áudio novo.",
        "pagina": "/som.html",
        "estado": "pronto",
        "prova": "não — som não tem resposta certa; quem julga é o seu ouvido",
        "vram_gb": 4,
        "ram_gb": 8,
        "modulos": ["torch", "transformers", "soundfile"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-som.txt",
        "nota": "Modelo pronto da Meta, licença CC-BY-NC: serve para testar e para uso "
                "interno, mas não para vender o áudio gerado. Sem GPU demora muito.",
    },
    {
        "id": "som-estilo",
        "nome": "Estilo de som (LoRA no MusicGen)",
        "faz": "Ensina um jeito de soar a partir das suas faixas.",
        "pagina": "/som.html",
        "estado": "pronto",
        "prova": "não — você ouve e diz se ficou parecido",
        "vram_gb": 6,
        "ram_gb": 12,
        "modulos": ["torch", "transformers", "soundfile", "peft", "scipy"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-som.txt",
        "nota": "As faixas viram fichas de áudio (EnCodec) e o LoRA aprende a continuar "
                "nesse estilo. Poucas faixas dão pouco resultado: junte pelo menos 20.",
    },
    {
        "id": "imagem-para-3d",
        "nome": "Texto ou foto → 3D (gerar malha)",
        "faz": "De uma frase ou de uma foto sai um .glb para abrir no Blender ou no motor de jogo.",
        "pagina": "/tresd.html",
        "estado": "pronto",
        "prova": "não — malha se julga olhando",
        "vram_gb": 6,
        "ram_gb": 12,
        "modulos": ["torch", "diffusers", "trimesh"],
        "programas": [],
        "instalar": "uv pip install --python .venv-train/bin/python -r requirements-3d.txt",
        "nota": "Não é treino: é rodar o Shap-E (OpenAI, Apache-2.0), que já vem no "
                "diffusers. A malha sai grosseira — serve de rascunho, não de peça final. "
                "O TripoSR gera melhor, mas compila código CUDA na instalação e quebra em "
                "máquina sem compilador C; por isso não é ele que está aqui.",
    },
    {
        "id": "fotos-para-3d",
        "nome": "Fotos → 3D (Gaussian Splatting)",
        "faz": "Dezenas de fotos de um objeto viram uma cópia 3D dele.",
        "pagina": "",
        "estado": "planejado",
        "prova": "não",
        "vram_gb": 8,
        "ram_gb": 16,
        "modulos": ["torch"],
        "programas": ["colmap"],
        "instalar": "sudo apt install colmap && uv pip install nerfstudio",
        "nota": "Reconstrói UM objeto; não gera nada novo. Exige o COLMAP instalado no sistema "
                "(precisa de root), que não há nesta máquina.",
    },
    {
        "id": "voz",
        "nome": "Clonar uma voz",
        "faz": "Alguns minutos de áudio e o modelo fala com aquela voz.",
        "pagina": "",
        "estado": "planejado",
        "prova": "parcial",
        "vram_gb": 6,
        "ram_gb": 8,
        "modulos": ["torch", "torchaudio"],
        "programas": ["ffmpeg"],
        "instalar": "uv pip install openvoice-cli (licença MIT) — evite XTTS, cuja licença proíbe uso comercial",
        "nota": "Para uma ferramenta que você vai vender, a licença do modelo de voz importa "
                "mais que a qualidade: o XTTS é não comercial. Não implementado ainda.",
    },
    {
        "id": "video",
        "nome": "Treinar geração de vídeo",
        "faz": "—",
        "pagina": "",
        "estado": "impossivel",
        "prova": "não",
        "vram_gb": 48,
        "ram_gb": 64,
        "modulos": [],
        "programas": [],
        "instalar": "",
        "nota": "Nem numa 3060 nem em dez: o treino de vídeo é feito em cluster. Dá para "
                "rodar modelos prontos, não treinar.",
    },
    {
        "id": "meshy",
        "nome": "Treinar um gerador 3D do zero (tipo Meshy)",
        "faz": "—",
        "pagina": "",
        "estado": "impossivel",
        "prova": "não",
        "vram_gb": 80,
        "ram_gb": 128,
        "modulos": [],
        "programas": [],
        "instalar": "",
        "nota": "Precisa de milhares de modelos 3D (o Objaverse tem ~10 TB) e centenas de "
                "horas de GPU de datacenter. O caminho viável é usar um modelo pronto.",
    },
]


PACOTES = {
    "texto-fatos": ["torch", "transformers>=4.44", "datasets", "peft", "trl", "accelerate",
                    "bitsandbytes"],
    "imagem-estilo": ["diffusers>=0.31", "torchvision", "pillow"],
    "imagem-personagem": ["diffusers>=0.31", "torchvision", "pillow"],
    "imagem-sdxl": ["diffusers>=0.31", "torchvision", "pillow"],
    "classificador-imagem": ["torchvision", "pillow"],
    "classificador-som": ["torchvision", "soundfile", "scipy", "numpy"],
    "som-gerar": ["transformers>=4.44", "soundfile", "scipy", "sentencepiece"],
    "som-estilo": ["transformers>=4.44", "soundfile", "scipy", "sentencepiece", "peft"],
    "imagem-para-3d": ["diffusers>=0.31", "trimesh", "pillow", "numpy"],
}

_instalacao: dict = {"state": "idle", "id": "", "linhas": []}


def instalacao_status() -> dict:
    return dict(_instalacao)


def instalar(treino_id: str) -> dict:
    """Baixa so o que aquele treino precisa. Nada e instalado sem voce pedir."""
    import subprocess
    import threading

    from agentepc.config import ROOT

    if _instalacao["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma instalacao rodando"}
    pacotes = PACOTES.get(treino_id)
    if not pacotes:
        item = next((i for i in CATALOGO if i["id"] == treino_id), None)
        motivo = (item or {}).get("instalar") or "esse treino nao se instala por aqui"
        return {"accepted": False, "reason": motivo}
    _instalacao.update({"state": "running", "id": treino_id, "linhas": []})

    def _diz(msg):
        _instalacao["linhas"].append(msg)
        del _instalacao["linhas"][:-60]

    def _go():
        try:
            python = ROOT / ".venv-train" / "bin" / "python"
            if not python.exists():
                raise RuntimeError("falta o ambiente .venv-train; use Ambiente > instalar aqui")
            uv = shutil.which("uv")
            cmd = ([uv, "pip", "install", "--python", str(python), *pacotes] if uv
                   else [str(python), "-m", "pip", "install", *pacotes])
            _diz("instalando: " + " ".join(pacotes))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors="replace")
            for linha in proc.stdout:
                linha = linha.strip()
                if linha:
                    _diz(linha[-160:])
            if proc.wait() != 0:
                raise RuntimeError("a instalacao falhou; veja as linhas acima")
            _instalacao["state"] = "done"
            _diz("pronto — reinicie a pagina para o treino aparecer liberado")
        except Exception as exc:
            _instalacao["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="instalar-capacidade", daemon=True).start()
    return {"accepted": True}


def _tem_modulo(nome: str) -> bool:
    try:
        return importlib.util.find_spec(nome) is not None
    except (ImportError, ValueError):
        return False


def avaliar() -> dict:
    """Confere o catalogo contra esta maquina."""
    gpu, ram = _gpu(), _ram_gb()
    vram = gpu.get("vram_gb", 0) if gpu["tem"] else 0
    saida = []
    for item in CATALOGO:
        faltam_mod = [m for m in item["modulos"] if not _tem_modulo(m)]
        faltam_prog = [p for p in item["programas"] if not shutil.which(p)]
        folga = 1.08
        vram_ok = item["vram_gb"] == 0 or vram * folga >= item["vram_gb"]
        ram_ok = ram * folga >= item["ram_gb"]

        if item["estado"] == "impossivel":
            estado, motivo = "impossivel", item["nota"]
        elif item["estado"] == "planejado":
            estado, motivo = "planejado", item["nota"]
        elif not ram_ok:
            estado, motivo = "hardware", f"precisa de {item['ram_gb']} GB de RAM; esta máquina tem {ram}"
        elif not vram_ok:
            estado = "hardware"
            motivo = (f"precisa de {item['vram_gb']} GB de VRAM; "
                      + (f"esta placa tem {vram}" if gpu["tem"] else "não há placa de vídeo aqui"))
        elif faltam_mod or faltam_prog:
            estado = "instalar"
            motivo = "falta: " + ", ".join(faltam_mod + faltam_prog)
        else:
            estado, motivo = "disponivel", ""
        saida.append({**item, "situacao": estado, "motivo": motivo,
                      "faltam": faltam_mod + faltam_prog,
                      # tem_pacote = da para instalar por aqui, aqui ou noutro servidor
                      "tem_pacote": bool(PACOTES.get(item["id"])),
                      "instalavel": bool(PACOTES.get(item["id"])) and estado == "instalar"})
    return {
        "maquina": {"vram_gb": vram, "ram_gb": ram, "gpu": gpu.get("nome", "")},
        "itens": saida,
    }


def pacotes_de(ids: list[str]) -> list[str]:
    """Junta os pacotes dos treinos escolhidos, sem repetir, na ordem pedida.

    Serve para instalar num servidor remoto: la nao existe requirements-*.txt a menos que
    o projeto tenha sido enviado, entao o script leva os nomes dos pacotes no corpo.
    """
    saida: list[str] = []
    for treino_id in ids:
        for pacote in PACOTES.get(treino_id, []):
            if pacote not in saida:
                saida.append(pacote)
    return saida


def nomes_de(ids: list[str]) -> str:
    achados = [i["nome"] for i in CATALOGO if i["id"] in ids]
    return ", ".join(achados)


def liberado(treino_id: str) -> tuple[bool, str]:
    """Usado pelas paginas antes de deixar treinar."""
    item = next((i for i in avaliar()["itens"] if i["id"] == treino_id), None)
    if not item:
        return False, "treino desconhecido"
    return item["situacao"] == "disponivel", item["motivo"]
