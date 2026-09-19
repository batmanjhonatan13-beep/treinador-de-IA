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
    "colecao": "", "saida": "", "amostras": [], "terminou": False, "tipo": "", "base": "",
}


def _diz(msg: str) -> None:
    _job["linhas"].append(msg)
    del _job["linhas"][:-120]


def status() -> dict:
    return {**_job, "modelos": treinados(), "base": base_id(), "gpu": _gpu_ok(),
            "bases": [{"chave": k, **v} for k, v in BASES.items()]}


BASES = {
    "sd15": {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "nome": "Stable Diffusion 1.5",
             "capacidade": "imagem-estilo", "sdxl": False},
    "sdxl": {"id": "stabilityai/stable-diffusion-xl-base-1.0", "nome": "SDXL 1.0",
             "capacidade": "imagem-sdxl", "sdxl": True},
}


def base_id(chave: str = "") -> str:
    if chave in BASES:
        return BASES[chave]["id"]
    return (load().get("imagem") or {}).get("base") or BASES["sd15"]["id"]


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
            resolucao: int = 512, lr: float = 1e-4, rank: int = 16,
            tipo: str = "estilo", base: str = "sd15") -> dict:
    from agentepc import capacidades

    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um treino de imagem rodando"}
    if base not in BASES:
        return {"accepted": False, "reason": "modelo base desconhecido"}
    # o catalogo decide se esta maquina aguenta; o botao nao promete o que ela nao faz
    cap = BASES[base]["capacidade"] if tipo == "estilo" else "imagem-personagem"
    if base == "sdxl":
        cap = "imagem-sdxl"
    ok, motivo = capacidades.liberado(cap)
    if not ok:
        return {"accepted": False, "reason": motivo or "treino nao disponivel nesta maquina"}
    pasta = pasta_imagens() / colecao
    imagens = [p for p in sorted(pasta.glob("*")) if p.suffix.lower() in EXT]
    if len(imagens) < 4:
        return {"accepted": False, "reason": f"poucas imagens ({len(imagens)}); junte pelo menos 4"}
    gatilho = (gatilho or colecao).strip()
    destino = saida_raiz() / (nome or f"{colecao}-{datetime.now().strftime('%Y%m%d-%H%M')}")
    destino.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "passo": 0, "total": int(passos),
                 "perda": 0.0, "colecao": colecao, "saida": str(destino), "amostras": [],
                 "terminou": False, "tipo": tipo, "base": base})

    def _go() -> None:
        try:
            _treino(imagens, gatilho, int(passos), destino, int(resolucao), float(lr), int(rank),
                    tipo, base)
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


def _treino(imagens, gatilho, passos, destino, resolucao, lr, rank,
            tipo="estilo", chave_base="sd15") -> None:
    import torch
    from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
                           StableDiffusionXLPipeline, UNet2DConditionModel)
    from peft import LoraConfig, get_peft_model_state_dict
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPTextModel, CLIPTokenizer

    xl = bool(BASES.get(chave_base, {}).get("sdxl"))
    base = base_id(chave_base)
    _diz(f"carregando {base} na GPU" + (" (SDXL: dois codificadores de texto)" if xl else ""))
    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    texto = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to("cuda", torch.float16)
    tok2 = texto2 = None
    if xl:
        from transformers import CLIPTextModelWithProjection

        tok2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
        texto2 = CLIPTextModelWithProjection.from_pretrained(
            base, subfolder="text_encoder_2").to("cuda", torch.float16)
        texto2.requires_grad_(False)
    # o VAE do SDXL estoura em fp16; em fp32 ele e pequeno e nao atrapalha
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(
        "cuda", torch.float32 if xl else torch.float16)
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
    # estilo: a legenda descreve a estetica. personagem: o gatilho nomeia o sujeito, e a
    # frase precisa dizer que aquilo e um sujeito, senao o modelo aprende o fundo junto.
    padrao = gatilho if tipo == "estilo" else f"uma foto de {gatilho}"
    amostras = []
    for img in imagens:
        try:
            with Image.open(img) as im:
                amostras.append((prep(im.convert("RGB")), mapa.get(img.name, padrao)))
        except Exception as exc:
            _diz(f"pulei {img.name}: {exc}")
    if not amostras:
        raise RuntimeError("nenhuma imagem pode ser lida")
    _diz(f"{len(amostras)} imagem(ns) em {resolucao}px · {tipo} · legenda base: \"{padrao}\"")

    otim = torch.optim.AdamW(treinaveis, lr=lr)
    unet.enable_gradient_checkpointing()
    escala = torch.amp.GradScaler("cuda")
    passo = 0
    while passo < passos and _job["state"] == "running":
        for pixels, legenda in amostras:
            if passo >= passos or _job["state"] != "running":
                break
            pixels = pixels.unsqueeze(0).to("cuda", torch.float32 if xl else torch.float16)
            extra = {}
            with torch.no_grad():
                latente = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                ids = tok(legenda, padding="max_length", max_length=tok.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to("cuda")
                if xl:
                    # SDXL junta o penultimo estado dos dois codificadores e ainda quer saber
                    # o tamanho e o recorte da imagem original
                    ids2 = tok2(legenda, padding="max_length", max_length=tok2.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to("cuda")
                    s1 = texto(ids, output_hidden_states=True).hidden_states[-2]
                    fora2 = texto2(ids2, output_hidden_states=True)
                    s2 = fora2.hidden_states[-2]
                    contexto = torch.cat([s1, s2], dim=-1)
                    extra = {
                        "text_embeds": fora2.text_embeds,
                        "time_ids": torch.tensor(
                            [[resolucao, resolucao, 0, 0, resolucao, resolucao]],
                            device="cuda", dtype=torch.float32),
                    }
                else:
                    contexto = texto(ids)[0]
            latente = latente.to(torch.float32)
            barulho = torch.randn_like(latente)
            t = torch.randint(0, ruido.config.num_train_timesteps, (1,), device="cuda").long()
            sujo = ruido.add_noise(latente, barulho, t)
            with torch.autocast("cuda", dtype=torch.float16):
                previsto = unet(sujo, t, encoder_hidden_states=contexto.to(torch.float32),
                                added_cond_kwargs={k: v.to(torch.float32) for k, v in extra.items()}
                                or None).sample
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
    salvador = StableDiffusionXLPipeline if xl else StableDiffusionPipeline
    salvador.save_lora_weights(str(destino), unet_lora_layers=pesos, safe_serialization=True)
    (destino / "treino.json").write_text(json.dumps({
        "colecao": Path(imagens[0]).parent.name, "gatilho": gatilho, "passos": passo,
        "tipo": tipo,
        "base": base, "base_chave": chave_base, "resolucao": resolucao, "rank": rank, "lr": lr,
        "imagens": len(amostras),
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aviso": "Estilo nao tem prova automatica: julgue pelas amostras.",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _diz(f"pesos salvos em {destino}")

    del otim, unet, vae, texto
    if texto2 is not None:
        del texto2
    torch.cuda.empty_cache()
    _amostras(base, destino, gatilho, tipo, xl)


def _amostras(base: str, destino: Path, gatilho: str, tipo: str = "estilo", xl: bool = False) -> None:
    """Gera duas imagens com o LoRA ligado, para voce julgar o resultado."""
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    _diz("gerando amostras")
    if xl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16").to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16,
                                                       safety_checker=None).to("cuda")
    pipe.load_lora_weights(str(destino))
    pedidos = ([f"{gatilho}, uma paisagem", f"{gatilho}, o retrato de uma pessoa"]
               if tipo == "estilo" else
               [f"uma foto de {gatilho} em um parque", f"uma foto de {gatilho}, close do rosto"])
    for i, pedido in enumerate(pedidos, 1):
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


# ------------------------------------------------------------------ gerar para testar
_geracao: dict = {"state": "idle", "linhas": [], "arquivo": "", "pedido": ""}


def geracao_status() -> dict:
    return dict(_geracao)


def galeria() -> Path:
    path = resolve("data/geradas")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gerar(pedido: str, estilo: str = "", passos: int = 25, forca: float = 0.9) -> dict:
    """Gera uma imagem com um estilo treinado — e assim que voce confere se ele pegou."""
    import threading

    if _geracao["state"] == "running" or _job["state"] == "running":
        return {"accepted": False, "reason": "a GPU esta ocupada com um treino ou outra imagem"}
    if not _gpu_ok():
        return {"accepted": False, "reason": "gerar imagem aqui precisa de GPU"}
    pedido = (pedido or "").strip()
    if not pedido:
        return {"accepted": False, "reason": "escreva o que voce quer ver"}
    pasta = (saida_raiz() / estilo) if estilo else None
    if estilo and not (pasta / "pytorch_lora_weights.safetensors").exists():
        return {"accepted": False, "reason": "esse estilo treinado nao foi encontrado"}
    _geracao.update({"state": "running", "linhas": [], "arquivo": "", "pedido": pedido})

    def _diz(msg):
        _geracao["linhas"].append(msg)
        del _geracao["linhas"][:-30]

    def _go():
        try:
            import torch
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

            chave = "sd15"
            if pasta and (pasta / "treino.json").exists():
                chave = json.loads((pasta / "treino.json").read_text(encoding="utf-8")).get(
                    "base_chave", "sd15")
            base = base_id(chave)
            xl = bool(BASES.get(chave, {}).get("sdxl"))
            _diz(f"carregando {base}")
            if xl:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    base, torch_dtype=torch.float16, variant="fp16").to("cuda")
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    base, torch_dtype=torch.float16, safety_checker=None).to("cuda")
            if pasta:
                pipe.load_lora_weights(str(pasta))
                pipe.fuse_lora(lora_scale=float(forca))
                _diz(f"estilo {estilo} ligado (força {forca})")
            img = pipe(pedido, num_inference_steps=int(passos), guidance_scale=7.0).images[0]
            alvo = galeria() / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            img.save(alvo)
            _geracao["arquivo"] = alvo.name
            _diz(f"pronto: {alvo.name}")
            del pipe
            torch.cuda.empty_cache()
            _geracao["state"] = "done"
        except Exception as exc:
            _geracao["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="gerar-imagem", daemon=True).start()
    return {"accepted": True}
