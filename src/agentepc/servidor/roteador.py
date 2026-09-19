#!/usr/bin/env python3
"""Roteador: um endereco so na frente de varios modelos.

Um pacote exportado tem pecas diferentes — texto, imagem, som, 3D, classificador — e cada
uma e um modelo separado, em cima de uma base diferente. Elas NAO viram um modelo so: sao
arquiteturas distintas. O que da para fazer, e o que este arquivo faz, e por um maestro na
frente: quem chama fala com um endereco unico, e o maestro decide qual peca acorda.

Tres regras que valem o codigo inteiro:

  1. Nada fica ligado a toa. A peca so carrega quando chega o primeiro pedido dela, e volta
     a dormir depois de um tempo parada (--manter). Numa maquina com uma placa de video so,
     deixar tudo carregado nao e opcao: nao cabe.
  2. Uma peca de GPU por vez. Antes de acordar uma, as outras que estao na placa saem. E
     por isso que a primeira chamada depois de um tempo demora alguns segundos — e trabalho
     honesto, nao travamento.
  3. Quem so precisa de CPU fica no seu canto e nao disputa a placa.

A conversa e no formato da OpenAI (/v1/chat/completions), entao qualquer ferramenta que
fale com a OpenAI fala com este servidor — inclusive pedindo imagem, som ou 3D, porque cada
peca aparece como um "modelo" na lista.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Em maquina sem compilador C o torch tenta compilar kernel no primeiro passo e quebra.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

AQUI = Path(__file__).resolve().parent
SYSTEM = "Responda curto, so o fato. Se nao souber, diga que nao sabe."

# onde cada tipo consegue rodar. "gpu" nao e preciosismo: difusao em CPU nao e lenta, e
# inviavel — uma imagem passa de meia hora.
ONDE = {
    "texto": {"gpu": "recomendado", "cpu": "funciona, devagar", "vram_gb": 6, "ram_gb": 8},
    "imagem": {"gpu": "obrigatorio", "cpu": "nao", "vram_gb": 6, "ram_gb": 8},
    "som-estilo": {"gpu": "recomendado", "cpu": "funciona, muito devagar", "vram_gb": 4, "ram_gb": 8},
    "som-classificador": {"gpu": "opcional", "cpu": "funciona bem", "vram_gb": 0, "ram_gb": 4},
    "classificador": {"gpu": "opcional", "cpu": "funciona bem", "vram_gb": 0, "ram_gb": 4},
    "3d": {"gpu": "obrigatorio", "cpu": "nao", "vram_gb": 6, "ram_gb": 12},
}

USA_GPU = {"texto", "imagem", "som-estilo", "3d"}      # disputam a placa


def _agora() -> float:
    return time.time()


def _tem_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class Peca:
    """Uma peca do pacote: sabe acordar, responder e voltar a dormir."""

    def __init__(self, info: dict, raiz: Path):
        self.tipo = info["tipo"]
        self.id = info["id"]
        self.base = info.get("modelo") or ""
        self.detalhe = info.get("detalhe") or ""
        self.pasta = raiz / self.tipo / self.id
        self.obj = None
        self.ultimo_uso = 0.0
        self.carregando = False
        self.forca = 0.9          # quanto do seu estilo entra na imagem

    # ------------------------------------------------------------ estado
    @property
    def ligada(self) -> bool:
        return self.obj is not None

    @property
    def usa_gpu(self) -> bool:
        return self.tipo in USA_GPU

    def estado(self) -> dict:
        return {
            "id": self.id, "tipo": self.tipo, "base": self.base,
            "ligada": self.ligada, "usa_gpu": self.usa_gpu,
            "onde": ONDE.get(self.tipo, {}),
            "parada_ha_s": round(_agora() - self.ultimo_uso) if self.ultimo_uso else None,
            "detalhe": self.detalhe,
        }

    # ------------------------------------------------------------ carga
    def carregar(self, diz) -> None:
        if self.obj is not None:
            return
        self.carregando = True
        try:
            inicio = _agora()
            diz(f"ligando {self.tipo}:{self.id}")
            self.obj = getattr(self, f"_carrega_{self.tipo.replace('-', '_')}")(diz)
            diz(f"{self.tipo}:{self.id} pronto em {_agora() - inicio:.0f}s")
        finally:
            self.carregando = False
        self.ultimo_uso = _agora()

    def soltar(self, diz) -> None:
        if self.obj is None:
            return
        diz(f"desligando {self.tipo}:{self.id}")
        self.obj = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------- carregadores
    def _carrega_texto(self, diz):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adaptador = self.pasta / "adapter"
        base = self.base or json.loads((self.pasta / "pacote.json").read_text())["base_hf"]
        local = self.pasta / "base"
        if (local / "config.json").exists():
            base = str(local)
        tok = AutoTokenizer.from_pretrained(
            str(adaptador) if (adaptador / "tokenizer.json").exists() else base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        extra = {}
        if torch.cuda.is_available():
            # 4 bits de proposito: o adaptador foi treinado sobre a base nesse formato, e
            # em precisao cheia as mesmas perguntas saem diferentes (medido no projeto)
            try:
                from transformers import BitsAndBytesConfig

                extra["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)
                extra["device_map"] = "auto"
            except Exception as exc:
                diz(f"sem bitsandbytes ({exc}); carregando em 16 bits")
                extra["dtype"] = torch.float16
                extra["device_map"] = "auto"
        else:
            diz("sem placa de video: texto vai rodar na memoria, mais devagar")
        modelo = AutoModelForCausalLM.from_pretrained(base, **extra)
        modelo = PeftModel.from_pretrained(modelo, str(adaptador))
        modelo.eval()
        return {"modelo": modelo, "tok": tok}

    def _carrega_imagem(self, diz):
        import torch
        from diffusers import DiffusionPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("geracao de imagem precisa de placa de video")
        meta = json.loads((self.pasta / "modelo" / "treino.json").read_text(encoding="utf-8"))
        base = meta.get("base_id") or meta.get("base") or "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = DiffusionPipeline.from_pretrained(base, dtype=torch.float16, safety_checker=None)
        pipe.load_lora_weights(str(self.pasta / "modelo"))
        # fundir na hora de carregar e o caminho que o projeto ja mediu funcionando; mudar a
        # forca depois exige recarregar, e o roteador faz isso sozinho quando voce pede outra
        pipe.fuse_lora(lora_scale=float(self.forca))
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return {"pipe": pipe, "gatilho": meta.get("gatilho", "")}

    def _carrega_som_estilo(self, diz):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        meta = json.loads((self.pasta / "modelo" / "treino.json").read_text(encoding="utf-8"))
        base = meta.get("base") or "facebook/musicgen-small"
        proc = AutoProcessor.from_pretrained(base)
        modelo = MusicgenForConditionalGeneration.from_pretrained(
            base, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        modelo.decoder = PeftModel.from_pretrained(modelo.decoder, str(self.pasta / "modelo"))
        modelo = modelo.to("cuda" if torch.cuda.is_available() else "cpu")
        return {"modelo": modelo, "proc": proc, "gatilho": meta.get("gatilho", "")}

    def _carrega_som_classificador(self, diz):
        import torch
        from torchvision import models

        dados = torch.load(self.pasta / "modelo" / "modelo.pt", map_location="cpu",
                           weights_only=False)
        rede = models.resnet18()
        rede.fc = torch.nn.Linear(rede.fc.in_features, len(dados["classes"]))
        rede.load_state_dict(dados["pesos"])
        rede.eval()
        return {"rede": rede, "classes": dados["classes"]}

    def _carrega_classificador(self, diz):
        import torch
        from torchvision import models

        dados = torch.load(self.pasta / "modelo" / "modelo.pt", map_location="cpu",
                           weights_only=False)
        rede = models.resnet18()
        rede.fc = torch.nn.Linear(rede.fc.in_features, len(dados["classes"]))
        rede.load_state_dict(dados["pesos"])
        rede.eval()
        return {"rede": rede, "classes": dados["classes"]}

    def _carrega_3d(self, diz):
        import torch
        from diffusers import ShapEPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("geracao 3D precisa de placa de video")
        pipe = ShapEPipeline.from_pretrained("openai/shap-e", dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return {"pipe": pipe}


class Maestro:
    """Decide quem acorda, quem dorme e quem responde."""

    def __init__(self, raiz: Path, manter: int = 300, verbose: bool = True):
        self.raiz = raiz
        self.manter = manter
        self.verbose = verbose
        self.linhas: list[str] = []
        self.trava = threading.Lock()
        dados = json.loads((raiz / "pacote.json").read_text(encoding="utf-8"))
        itens = dados.get("itens") or [dados]
        self.pecas = {}
        for info in itens:
            if not info.get("tipo") or not info.get("id"):
                # pacote antigo, de antes do roteador: nao da para saber que peca e essa
                self.diz("pacote.json sem 'tipo'/'id': este pacote e anterior ao roteador")
                continue
            peca = Peca(info, raiz)
            if peca.pasta.is_dir():
                self.pecas[f"{peca.tipo}:{peca.id}"] = peca
            else:
                self.diz(f"aviso: {peca.tipo}:{peca.id} esta no pacote.json mas a pasta nao existe")
        threading.Thread(target=self._vigia, name="soneca", daemon=True).start()

    def diz(self, msg: str) -> None:
        linha = time.strftime("%H:%M:%S ") + msg
        self.linhas.append(linha)
        del self.linhas[:-200]
        if self.verbose:
            print("[roteador]", msg, flush=True)

    # ------------------------------------------------------------ escolha
    def achar(self, nome: str) -> Peca | None:
        """Aceita "tipo:id", so o id, ou so o tipo (pega a primeira daquele tipo)."""
        if nome in self.pecas:
            return self.pecas[nome]
        for chave, peca in self.pecas.items():
            if nome in (peca.id, peca.tipo) or chave.endswith(f":{nome}"):
                return peca
        return None

    def usar(self, peca: Peca):
        """Garante a peca ligada, tirando da placa quem estiver atrapalhando."""
        if peca.usa_gpu and _tem_cuda():
            for outra in self.pecas.values():
                if outra is not peca and outra.ligada and outra.usa_gpu:
                    self.diz(f"a placa e uma so: {outra.tipo}:{outra.id} sai para {peca.tipo}:{peca.id} entrar")
                    outra.soltar(self.diz)
        peca.carregar(self.diz)
        peca.ultimo_uso = _agora()
        return peca.obj

    def _vigia(self) -> None:
        """Quem ficou parado tempo demais volta a dormir e devolve a memoria."""
        while True:
            time.sleep(20)
            if self.manter <= 0:
                continue
            for peca in list(self.pecas.values()):
                if peca.ligada and not peca.carregando and peca.ultimo_uso:
                    parada = _agora() - peca.ultimo_uso
                    if parada > self.manter:
                        with self.trava:
                            if peca.ligada and _agora() - peca.ultimo_uso > self.manter:
                                self.diz(f"{peca.tipo}:{peca.id} parada ha {parada:.0f}s")
                                peca.soltar(self.diz)

    # ------------------------------------------------------------ trabalho
    def responder(self, peca: Peca, mensagens: list[dict], limite: int = 160) -> str:
        import torch

        with self.trava:
            pronto = self.usar(peca)
            tok, modelo = pronto["tok"], pronto["modelo"]
            sistema = next((m["content"] for m in mensagens if m.get("role") == "system"), SYSTEM)
            pergunta = next((m["content"] for m in reversed(mensagens)
                             if m.get("role") == "user"), "")
            # o formato tem que ser este: foi com ele que o treino aconteceu, e o
            # conhecimento so e acionado quando a pergunta chega do mesmo jeito
            texto = f"system: {sistema}\nuser: {pergunta}\nassistant:"
            entrada = tok(texto, return_tensors="pt").to(modelo.device)
            with torch.no_grad():
                saida = modelo.generate(**entrada, max_new_tokens=limite, do_sample=False,
                                        pad_token_id=tok.pad_token_id)
            resposta = tok.decode(saida[0][entrada["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
            return resposta.strip()

    def imagem(self, peca: Peca, pedido: str, passos: int = 25, forca: float = 0.9) -> bytes:
        import io

        with self.trava:
            if peca.ligada and abs(peca.forca - forca) > 0.01:
                self.diz(f"forca mudou ({peca.forca} -> {forca}): recarregando o estilo")
                peca.soltar(self.diz)
            peca.forca = forca
            pronto = self.usar(peca)
            gatilho = pronto["gatilho"]
            frase = f"{gatilho}, {pedido}" if gatilho and gatilho not in pedido else pedido
            imagem = pronto["pipe"](frase, num_inference_steps=passos).images[0]
            buf = io.BytesIO()
            imagem.save(buf, format="PNG")
            return buf.getvalue()

    def som(self, peca: Peca, pedido: str, segundos: int = 8) -> bytes:
        import io

        import soundfile as sf
        import torch

        with self.trava:
            pronto = self.usar(peca)
            modelo, proc = pronto["modelo"], pronto["proc"]
            entrada = proc(text=[pedido], padding=True, return_tensors="pt").to(modelo.device)
            with torch.no_grad():
                onda = modelo.generate(**entrada, do_sample=True, guidance_scale=3.0,
                                       max_new_tokens=int(segundos) * 50)
            buf = io.BytesIO()
            sf.write(buf, onda[0, 0].float().cpu().numpy(),
                     modelo.config.audio_encoder.sampling_rate, format="WAV")
            return buf.getvalue()

    def malha(self, peca: Peca, pedido: str, passos: int = 64) -> bytes:
        import tempfile

        from diffusers.utils import export_to_ply

        with self.trava:
            pronto = self.usar(peca)
            saida = pronto["pipe"](pedido, guidance_scale=15.0, num_inference_steps=passos,
                                   frame_size=256, output_type="mesh").images
            tmp = Path(tempfile.mkstemp(suffix=".ply")[1])
            export_to_ply(saida[0], str(tmp))
            try:
                import trimesh

                glb = tmp.with_suffix(".glb")
                trimesh.load(str(tmp)).export(str(glb))
                dados = glb.read_bytes()
                glb.unlink(missing_ok=True)
            except Exception:
                dados = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            return dados

    def classificar(self, peca: Peca, arquivo: bytes, nome: str = "") -> dict:
        import tempfile

        import torch

        with self.trava:
            pronto = self.usar(peca)
            rede, classes = pronto["rede"], pronto["classes"]
            sufixo = Path(nome).suffix or (".wav" if peca.tipo == "som-classificador" else ".png")
            tmp = Path(tempfile.mkstemp(suffix=sufixo)[1])
            tmp.write_bytes(arquivo)
            try:
                if peca.tipo == "som-classificador":
                    votos = self._votos_som(rede, classes, tmp)
                else:
                    votos = self._votos_imagem(rede, classes, tmp)
            finally:
                tmp.unlink(missing_ok=True)
            ordem = sorted(votos.items(), key=lambda kv: kv[1], reverse=True)
            return {"resposta": ordem[0][0], "certeza": round(ordem[0][1], 3),
                    "todas": {k: round(v, 3) for k, v in ordem}}

    def _votos_imagem(self, rede, classes, caminho: Path) -> dict:
        import torch
        from PIL import Image
        from torchvision import transforms

        prep = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with Image.open(caminho) as im:
            x = prep(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(rede(x)[0], dim=0)
        return {c: float(p[i]) for i, c in enumerate(classes)}

    def _votos_som(self, rede, classes, caminho: Path) -> dict:
        import torch

        from som_preparo import fatias, espectro      # vai junto no pacote

        votos = torch.zeros(len(classes))
        pedacos = fatias(caminho, maximo=10)
        if not pedacos:
            raise RuntimeError("audio curto demais ou em silencio")
        with torch.no_grad():
            for onda in pedacos:
                mel = espectro(onda).unsqueeze(0).unsqueeze(0)
                mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode="bilinear",
                                                      align_corners=False)[0].repeat(3, 1, 1)
                votos += torch.softmax(rede(mel.unsqueeze(0))[0], dim=0)
        votos = votos / votos.sum()
        return {c: float(votos[i]) for i, c in enumerate(classes)}


# ---------------------------------------------------------------- servidor HTTP
class Handler(BaseHTTPRequestHandler):
    maestro: Maestro = None       # preenchido no main()
    server_version = "roteador-agente-pc"

    def log_message(self, fmt: str, *args) -> None:
        pass          # o maestro ja fala o que importa

    # ------------------------------------------------------------ utilidades
    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _erro(self, code: int, msg: str) -> None:
        # formato de erro da OpenAI: as ferramentas sabem ler este
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

    def _corpo(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        bruto = self.rfile.read(n)
        try:
            return json.loads(bruto.decode())
        except ValueError:
            return {"_bruto": bruto}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ------------------------------------------------------------ GET
    def do_GET(self) -> None:
        caminho = self.path.split("?")[0]
        m = self.maestro
        if caminho in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": chave, "object": "model", "owned_by": "voce",
                 "created": int(peca.ultimo_uso or 0), "tipo": peca.tipo,
                 "base": peca.base, "onde": ONDE.get(peca.tipo, {})}
                for chave, peca in m.pecas.items()]})
            return
        if caminho == "/status":
            self._json(200, {
                "pecas": [p.estado() for p in m.pecas.values()],
                "gpu": _tem_cuda(),
                "manter_ligado_s": m.manter,
                "linhas": m.linhas[-30:],
            })
            return
        if caminho == "/saude":
            self._json(200, {"ok": True, "pecas": len(m.pecas)})
            return
        self._erro(404, f"nao existe {caminho}. Veja API.md ou GET /status")

    # ------------------------------------------------------------ POST
    def do_POST(self) -> None:
        caminho = self.path.split("?")[0]
        m = self.maestro
        try:
            corpo = self._corpo()
            if caminho in ("/v1/chat/completions", "/chat/completions"):
                self._conversa(corpo)
                return
            if caminho in ("/v1/images/generations", "/gerar/imagem"):
                peca = m.achar(corpo.get("model") or "imagem")
                if not peca or peca.tipo != "imagem":
                    self._erro(404, "este pacote nao tem peca de imagem")
                    return
                png = m.imagem(peca, corpo.get("prompt") or "", int(corpo.get("passos") or 25),
                               float(corpo.get("forca") or 0.9))
                self._json(200, {"created": int(_agora()), "data": [
                    {"b64_json": base64.b64encode(png).decode()}]})
                return
            if caminho in ("/v1/audio/generations", "/gerar/som"):
                peca = m.achar(corpo.get("model") or "som-estilo")
                if not peca or peca.tipo != "som-estilo":
                    self._erro(404, "este pacote nao tem peca de som")
                    return
                wav = m.som(peca, corpo.get("prompt") or "", int(corpo.get("segundos") or 8))
                self._json(200, {"created": int(_agora()),
                                 "data": [{"b64_wav": base64.b64encode(wav).decode()}]})
                return
            if caminho in ("/v1/3d/generations", "/gerar/3d"):
                peca = m.achar(corpo.get("model") or "3d")
                if not peca or peca.tipo != "3d":
                    self._erro(404, "este pacote nao tem peca de 3D")
                    return
                glb = m.malha(peca, corpo.get("prompt") or "", int(corpo.get("passos") or 64))
                self._json(200, {"created": int(_agora()),
                                 "data": [{"b64_glb": base64.b64encode(glb).decode()}]})
                return
            if caminho == "/classificar":
                peca = m.achar(corpo.get("model") or "")
                if not peca or "classificador" not in peca.tipo:
                    self._erro(404, "diga qual classificador em \"model\" (veja GET /v1/models)")
                    return
                bruto = corpo.get("arquivo_b64")
                if not bruto:
                    self._erro(400, "mande o arquivo em \"arquivo_b64\" (base64)")
                    return
                self._json(200, m.classificar(peca, base64.b64decode(bruto),
                                              corpo.get("nome") or ""))
                return
            if caminho in ("/ligar", "/desligar"):
                peca = m.achar(corpo.get("model") or "")
                if not peca:
                    self._erro(404, "peca desconhecida")
                    return
                with m.trava:
                    if caminho == "/ligar":
                        m.usar(peca)
                    else:
                        peca.soltar(m.diz)
                self._json(200, peca.estado())
                return
            self._erro(404, f"nao existe {caminho}. Veja API.md ou GET /status")
        except Exception as exc:
            m.diz(f"erro em {caminho}: {exc}")
            self._erro(500, str(exc))

    # ------------------------------------------------------------ conversa
    def _conversa(self, corpo: dict) -> None:
        """Uma porta so para tudo: o "model" escolhido decide o que acontece.

        Isso existe para ferramenta que so sabe conversar (Open WebUI, Continue, qualquer
        cliente da OpenAI) conseguir pedir imagem, som ou 3D: e so escolher a peca na lista
        de modelos e mandar a mensagem. A resposta volta como markdown com o arquivo dentro.
        """
        m = self.maestro
        mensagens = corpo.get("messages") or []
        pedido_modelo = corpo.get("model") or ""
        peca = m.achar(pedido_modelo) or next(
            (p for p in m.pecas.values() if p.tipo == "texto"), None)
        if not peca:
            self._erro(404, f"nao achei a peca \"{pedido_modelo}\"; veja GET /v1/models")
            return
        pergunta = next((x.get("content", "") for x in reversed(mensagens)
                         if x.get("role") == "user"), "")
        if peca.tipo == "texto":
            texto = m.responder(peca, mensagens, int(corpo.get("max_tokens") or 160))
        elif peca.tipo == "imagem":
            png = m.imagem(peca, pergunta)
            texto = f"![imagem](data:image/png;base64,{base64.b64encode(png).decode()})"
        elif peca.tipo == "som-estilo":
            wav = m.som(peca, pergunta)
            texto = ("<audio controls src=\"data:audio/wav;base64,"
                     + base64.b64encode(wav).decode() + "\"></audio>")
        elif peca.tipo == "3d":
            glb = m.malha(peca, pergunta)
            texto = (f"Malha pronta ({len(glb) / 1e6:.1f} MB). Em base64 o .glb nao serve para "
                     "ver aqui — use POST /v1/3d/generations e salve o arquivo.")
        else:
            texto = ("Esta peca e um classificador: mande o arquivo em POST /classificar. "
                     f"Categorias: {', '.join(peca.obj['classes']) if peca.ligada else 'ligue a peca para ver'}")
        resposta = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(_agora()),
            "model": f"{peca.tipo}:{peca.id}",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": texto}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        if not corpo.get("stream"):
            self._json(200, resposta)
            return
        # Streaming: a resposta sai inteira de uma vez, empacotada como SSE. Muitos clientes
        # so aceitam stream; fingir o formato e honesto desde que esteja escrito na doc.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        pedaco = {"id": resposta["id"], "object": "chat.completion.chunk",
                  "created": resposta["created"], "model": resposta["model"],
                  "choices": [{"index": 0, "delta": {"role": "assistant", "content": texto},
                               "finish_reason": None}]}
        fim = {**pedaco, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        for parte in (pedaco, fim):
            self.wfile.write(f"data: {json.dumps(parte, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Roteador do pacote: um endereco, varios modelos")
    p.add_argument("--porta", type=int, default=8770)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--manter", type=int, default=300,
                   help="segundos parada antes de a peca dormir (0 = nunca dormir)")
    p.add_argument("--acordar", default="",
                   help="pecas para ja deixar ligadas ao subir (ex.: texto)")
    args = p.parse_args()

    maestro = Maestro(AQUI, manter=args.manter)
    Handler.maestro = maestro
    print(f"\nRoteador em http://{args.host}:{args.porta}")
    print("  compativel com a OpenAI:  POST /v1/chat/completions")
    print("  o que existe:             GET  /v1/models")
    print("  quem esta ligado:         GET  /status")
    print("  documentacao completa:    API.md, ao lado deste arquivo\n")
    for peca in maestro.pecas.values():
        onde = ONDE.get(peca.tipo, {})
        print(f"  {peca.tipo:18} {peca.id:24} GPU: {onde.get('gpu', '?'):12} "
              f"CPU: {onde.get('cpu', '?')}")
    if not _tem_cuda():
        print("\n  SEM PLACA DE VIDEO: imagem e 3D nao vao rodar aqui; texto e som vao, devagar.")
    print(f"\n  pecas dormem depois de {args.manter}s paradas" if args.manter
          else "\n  --manter 0: nada dorme sozinho")
    for nome in [x for x in args.acordar.split(",") if x.strip()]:
        peca = maestro.achar(nome.strip())
        if peca:
            with maestro.trava:
                maestro.usar(peca)
    ThreadingHTTPServer((args.host, args.porta), Handler).serve_forever()


if __name__ == "__main__":
    main()
