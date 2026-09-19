"""Treina um LoRA de estilo em cima do Stable Diffusion, aqui na sua placa.

E a mesma tecnica do treino de texto — poucos pesos novos em cima de um modelo pronto —
mas com uma diferenca que muda tudo: **nao existe prova automatica**. No texto da para
perguntar e conferir se a resposta bate. Em imagem nao ha resposta certa; quem julga se o
estilo pegou e voce, olhando as amostras. Por isso aqui nao ha selo verde.

So com GPU. Treino de difusao na CPU nao e lento, e inviavel.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from agentepc.config import load, resolve
from agentepc.coletor import raiz as pasta_imagens

EXT = (".jpg", ".jpeg", ".png", ".webp")

_job: dict = {
    "state": "idle", "linhas": [], "passo": 0, "total": 0, "perda": 0.0,
    "colecao": "", "saida": "", "amostras": [], "terminou": False,
}


def _diz(msg: str) -> None:
    _job["linhas"].append(msg)
    del _job["linhas"][:-120]


def status() -> dict:
    return {**_job, "modelos": treinados(), "base": base_id(), "gpu": _gpu_ok()}


def base_id() -> str:
    return (load().get("imagem") or {}).get("base") or "stable-diffusion-v1-5/stable-diffusion-v1-5"


def saida_raiz() -> Path:
    path = resolve("data/imagem-lora")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _gpu_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def treinados() -> list[dict]:
    saida = []
    for pasta in sorted(saida_raiz().iterdir()) if saida_raiz().exists() else []:
        if not pasta.is_dir():
            continue
        meta = pasta / "treino.json"
        dados = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        pesos = pasta / "pytorch_lora_weights.safetensors"
        saida.append({
            "id": pasta.name, **dados,
            "bytes": pesos.stat().st_size if pesos.exists() else 0,
            "amostras": [p.name for p in sorted(pasta.glob("amostra*.png"))],
            "pronto": pesos.exists(),
        })
    return saida


def legendas(colecao: str) -> dict:
    """Uma legenda por imagem. Sem arquivo proprio, usa a legenda geral do estilo."""
    pasta = pasta_imagens() / colecao
    arq = pasta / "legendas.txt"
    mapa = {}
    if arq.exists():
        for linha in arq.read_text(encoding="utf-8").splitlines():
            if "\t" in linha:
                nome, texto = linha.split("\t", 1)
                mapa[nome.strip()] = texto.strip()
    return mapa


def treinar(colecao: str, gatilho: str = "", passos: int = 600, nome: str = "",
            resolucao: int = 512, lr: float = 1e-4, rank: int = 16) -> dict:
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um treino de imagem rodando"}
    if not _gpu_ok():
        return {"accepted": False, "reason": "treino de imagem precisa de GPU; na CPU e inviavel"}
    pasta = pasta_imagens() / colecao
    imagens = [p for p in sorted(pasta.glob("*")) if p.suffix.lower() in EXT]
    if len(imagens) < 4:
        return {"accepted": False, "reason": f"poucas imagens ({len(imagens)}); junte pelo menos 4"}
    gatilho = (gatilho or colecao).strip()
    destino = saida_raiz() / (nome or f"{colecao}-{datetime.now().strftime('%Y%m%d-%H%M')}")
    destino.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "passo": 0, "total": int(passos),
                 "perda": 0.0, "colecao": colecao, "saida": str(destino), "amostras": [],
                 "terminou": False})

    def _go() -> None:
        try:
            _treino(imagens, gatilho, int(passos), destino, int(resolucao), float(lr), int(rank))
            _job["state"] = "done"
            _job["terminou"] = True
            _diz("TREINO CONCLUIDO — olhe as amostras e diga se o estilo pegou")
        except Exception as exc:
            _job["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="imagem-lora", daemon=True).start()
    return {"accepted": True, "saida": str(destino), "imagens": len(imagens)}


def parar() -> dict:
    if _job["state"] == "running":
        _job["state"] = "parando"
        _diz("parando no fim do passo atual")
    return status()


def _treino(imagens, gatilho, passos, destino, resolucao, lr, rank) -> None:
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model_state_dict
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPTextModel, CLIPTokenizer

    base = base_id()
    _diz(f"carregando {base} na GPU")
    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    texto = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to("cuda", torch.float16)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to("cuda", torch.float16)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to("cuda", torch.float32)
    ruido = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    texto.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # so a atencao: e nela que o estilo pega, e mantem o LoRA pequeno
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    treinaveis = [p for p in unet.parameters() if p.requires_grad]
    _diz(f"{sum(p.numel() for p in treinaveis) / 1e6:.1f}M pesos treinaveis (rank {rank})")

    prep = transforms.Compose([
        transforms.Resize(resolucao, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolucao),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mapa = legendas(Path(imagens[0]).parent.name)
    amostras = []
    for img in imagens:
        try:
            with Image.open(img) as im:
                amostras.append((prep(im.convert("RGB")), mapa.get(img.name, gatilho)))
        except Exception as exc:
            _diz(f"pulei {img.name}: {exc}")
    if not amostras:
        raise RuntimeError("nenhuma imagem pode ser lida")
    _diz(f"{len(amostras)} imagem(ns) prontas em {resolucao}px; legenda base: \"{gatilho}\"")

    otim = torch.optim.AdamW(treinaveis, lr=lr)
    unet.enable_gradient_checkpointing()
    escala = torch.amp.GradScaler("cuda")
    passo = 0
    while passo < passos and _job["state"] == "running":
        for pixels, legenda in amostras:
            if passo >= passos or _job["state"] != "running":
                break
            pixels = pixels.unsqueeze(0).to("cuda", torch.float16)
            with torch.no_grad():
                latente = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                ids = tok(legenda, padding="max_length", max_length=tok.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to("cuda")
                contexto = texto(ids)[0]
            latente = latente.to(torch.float32)
            barulho = torch.randn_like(latente)
            t = torch.randint(0, ruido.config.num_train_timesteps, (1,), device="cuda").long()
            sujo = ruido.add_noise(latente, barulho, t)
            with torch.autocast("cuda", dtype=torch.float16):
                previsto = unet(sujo, t, encoder_hidden_states=contexto.to(torch.float32)).sample
                perda = torch.nn.functional.mse_loss(previsto.float(), barulho.float())
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
            otim.zero_grad(set_to_none=True)
            passo += 1
            _job["passo"], _job["perda"] = passo, float(perda.detach())
            if passo % 50 == 0:
                _diz(f"passo {passo}/{passos} — perda {_job['perda']:.4f}")

    pesos = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(str(destino), unet_lora_layers=pesos,
                                              safe_serialization=True)
    (destino / "treino.json").write_text(json.dumps({
        "colecao": Path(imagens[0]).parent.name, "gatilho": gatilho, "passos": passo,
        "base": base, "resolucao": resolucao, "rank": rank, "lr": lr,
        "imagens": len(amostras),
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aviso": "Estilo nao tem prova automatica: julgue pelas amostras.",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _diz(f"pesos salvos em {destino}")

    del otim, unet, vae, texto
    torch.cuda.empty_cache()
    _amostras(base, destino, gatilho)


def _amostras(base: str, destino: Path, gatilho: str) -> None:
    """Gera duas imagens com o LoRA ligado, para voce julgar o resultado."""
    import torch
    from diffusers import StableDiffusionPipeline

    _diz("gerando amostras")
    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
    pipe.load_lora_weights(str(destino))
    for i, pedido in enumerate([f"{gatilho}, uma paisagem", f"{gatilho}, o retrato de uma pessoa"], 1):
        img = pipe(pedido, num_inference_steps=25, guidance_scale=7.0,
                   generator=torch.Generator("cuda").manual_seed(1000 + i)).images[0]
        img.save(destino / f"amostra{i}.png")
        _job["amostras"] = [p.name for p in sorted(destino.glob("amostra*.png"))]
        _diz(f"amostra {i}: {pedido}")
    del pipe
    torch.cuda.empty_cache()


def apagar(nome: str) -> dict:
    import shutil

    alvo = saida_raiz() / nome
    if not alvo.is_dir():
        return {"ok": False, "reason": "treino nao encontrado"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "modelos": treinados()}
