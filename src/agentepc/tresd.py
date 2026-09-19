"""3D: gerar malha a partir de texto ou de uma foto, e juntar referencias com licenca.

Aqui vale separar duas coisas que se confundem:

  * GERAR 3D roda um modelo pronto (o Shap-E, da OpenAI, Apache-2.0). Nao ha treino:
    voce descreve ou mostra uma foto, e sai um arquivo .glb para abrir no Blender ou
    jogar no motor de jogo. Quem tem GPU consegue em segundos.
  * TREINAR um gerador 3D do zero nao cabe em PC nenhum — sao milhares de modelos 3D e
    centenas de horas de GPU de datacenter. Isso esta no catalogo como impossivel, com o
    motivo escrito, e nao vai virar botao.

O que da para treinar de verdade e o lado da imagem: um LoRA que desenha o seu objeto
sempre igual, e dali sai a foto que vira malha. Por isso a pagina liga as duas coisas.

Escolhi o Shap-E e nao o TripoSR (que gera malha melhor) por um motivo pratico: o TripoSR
depende do torchmcubes, que compila codigo CUDA na instalacao. Numa maquina de cliente sem
compilador C aquilo quebra no pip, e o cliente fica sem nada. O Shap-E ja vem dentro do
diffusers e roda em qualquer lugar onde o treino de imagem roda.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from agentepc.coletor import abrir, json_de
from agentepc.config import resolve
from agentepc.lotes import slug

TEXTO_3D = "openai/shap-e"
FOTO_3D = "openai/shap-e-img2img"
POLYHAVEN = "https://api.polyhaven.com"

_job: dict = {"state": "idle", "linhas": [], "pedido": "", "saida": "", "arquivos": []}
_coleta: dict = {"state": "idle", "linhas": [], "baixados": 0, "alvo": 0, "pasta": ""}


def _diz(msg: str) -> None:
    _job["linhas"].append(msg)
    del _job["linhas"][:-60]


def _diz_coleta(msg: str) -> None:
    _coleta["linhas"].append(msg)
    del _coleta["linhas"][:-60]


def raiz() -> Path:
    path = resolve("data/modelos-3d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def referencias_raiz() -> Path:
    path = resolve("data/referencias-3d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def modelos() -> list[dict]:
    saida = []
    for pasta in sorted(raiz().iterdir(), reverse=True) if raiz().exists() else []:
        meta = pasta / "modelo.json"
        if meta.is_file():
            dados = json.loads(meta.read_text(encoding="utf-8"))
            saida.append({"id": pasta.name, **dados,
                          "arquivos": sorted(p.name for p in pasta.glob("*")
                                             if p.suffix in (".glb", ".ply", ".obj", ".png"))})
    return saida


def referencias() -> list[dict]:
    saida = []
    for pasta in sorted(referencias_raiz().iterdir()) if referencias_raiz().exists() else []:
        if not pasta.is_dir():
            continue
        meta = pasta / "creditos.json"
        dados = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        saida.append({"id": pasta.name, "itens": len(dados.get("itens") or []),
                      "termo": dados.get("termo", ""),
                      "bytes": sum(p.stat().st_size for p in pasta.rglob("*") if p.is_file())})
    return saida


def status() -> dict:
    return {"geracao": {**_job, "modelos": modelos()},
            "coleta": {**_coleta, "referencias": referencias()}}


# ---------------------------------------------------------------- previa
def _previa(caminho_ply: Path, destino: Path, lado: int = 384) -> bool:
    """Desenha a malha numa imagem sem precisar de placa de video nem OpenGL.

    Um render de verdade pediria uma janela grafica, que nao existe num servidor. Entao
    projeto os vertices na tela e guardo o mais perto da camera em cada pixel — da para
    ver a forma e a cor, que e o que voce precisa para decidir se presta.
    """
    try:
        import numpy as np
        from PIL import Image
        import trimesh

        malha = trimesh.load(str(caminho_ply))
        v = np.asarray(malha.vertices, dtype="float32")
        if not len(v):
            return False
        cores = getattr(malha.visual, "vertex_colors", None)
        cores = (np.asarray(cores, dtype="uint8")[:, :3] if cores is not None
                 else np.full((len(v), 3), 200, dtype="uint8"))
        # 3/4 de perfil: de frente, uma malha fica chapada e nao da para julgar nada
        ay, ax = np.deg2rad(35), np.deg2rad(-18)
        giro_y = np.array([[np.cos(ay), 0.0, np.sin(ay)],
                           [0.0, 1.0, 0.0],
                           [-np.sin(ay), 0.0, np.cos(ay)]], dtype="float32")
        giro_x = np.array([[1.0, 0.0, 0.0],
                           [0.0, np.cos(ax), -np.sin(ax)],
                           [0.0, np.sin(ax), np.cos(ax)]], dtype="float32")
        p = v @ (giro_x @ giro_y).T
        p -= p.min(0)
        p = p / float(p.max() or 1.0)
        xs = (p[:, 0] * (lado - 24) + 12).astype(int)
        ys = ((1 - p[:, 1]) * (lado - 24) + 12).astype(int)
        z = p[:, 2]
        # quem esta na frente aparece mais claro: sem isso a malha vira uma mancha so
        luz = (0.45 + 0.75 * (1.0 - z)).reshape(-1, 1)
        cores = np.clip(cores.astype("float32") * luz, 0, 255).astype("uint8")
        tela = np.full((lado, lado, 3), 18, dtype="uint8")
        prof = np.full((lado, lado), 1e9, dtype="float32")
        for i in range(len(v)):
            x, y = xs[i], ys[i]
            for dx in (0, 1):
                for dy in (0, 1):     # ponto de 2x2: um pixel so deixa a previa rala
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < lado and 0 <= yy < lado and z[i] < prof[yy, xx]:
                        prof[yy, xx] = z[i]
                        tela[yy, xx] = cores[i]
        Image.fromarray(tela).save(destino)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- gerar
def gerar(pedido: str, foto: str = "", passos: int = 64, guia: float = 15.0,
          nome: str = "") -> dict:
    """De uma frase ou de uma foto sai um .glb. Com foto, usa o Shap-E img2img."""
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma geracao 3D rodando"}
    from agentepc import capacidades

    ok, motivo = capacidades.liberado("imagem-para-3d")
    if not ok:
        return {"accepted": False, "reason": motivo}
    pedido = (pedido or "").strip()
    foto = (foto or "").strip()
    if not pedido and not foto:
        return {"accepted": False, "reason": "escreva o que quer, ou aponte uma foto"}
    destino = raiz() / (slug(nome) or slug(pedido)[:40]
                        or datetime.now().strftime("%Y%m%d-%H%M%S"))
    destino.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "pedido": pedido or foto,
                 "saida": str(destino), "arquivos": []})

    def _go() -> None:
        try:
            _gerar(pedido, foto, int(passos), float(guia), destino)
            _job["state"] = "done"
        except Exception as exc:
            _job["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="gerar-3d", daemon=True).start()
    return {"accepted": True, "saida": str(destino)}


def _gerar(pedido: str, foto: str, passos: int, guia: float, destino: Path) -> None:
    import torch
    from diffusers import ShapEImg2ImgPipeline, ShapEPipeline
    from diffusers.utils import export_to_ply

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tipo = torch.float16 if dispositivo == "cuda" else torch.float32
    inicio = datetime.now()

    def _passou() -> int:
        return (datetime.now() - inicio).seconds

    if foto:
        from PIL import Image

        caminho = Path(foto)
        if not caminho.is_absolute():
            caminho = resolve("data") / foto
        if not caminho.is_file():
            raise RuntimeError(f"nao achei a foto {foto}")
        _diz(f"[{_passou():4}s] carregando {FOTO_3D} (foto → 3D)")
        pipe = ShapEImg2ImgPipeline.from_pretrained(FOTO_3D, dtype=tipo).to(dispositivo)
        with Image.open(caminho) as im:
            entrada = im.convert("RGB").resize((256, 256))
        _diz(f"[{_passou():4}s] gerando a partir de {caminho.name}, {passos} passos")
        saida = pipe(entrada, guidance_scale=3.0, num_inference_steps=passos,
                     frame_size=256, output_type="mesh").images
        origem = f"foto: {caminho.name}"
    else:
        _diz(f"[{_passou():4}s] carregando {TEXTO_3D} (texto → 3D)")
        pipe = ShapEPipeline.from_pretrained(TEXTO_3D, dtype=tipo).to(dispositivo)
        _diz(f"[{_passou():4}s] gerando \"{pedido}\", {passos} passos")
        saida = pipe(pedido, guidance_scale=guia, num_inference_steps=passos,
                     frame_size=256, output_type="mesh").images
        origem = f"texto: {pedido}"

    ply = destino / "modelo.ply"
    export_to_ply(saida[0], str(ply))
    _diz(f"[{_passou():4}s] malha salva")

    formatos = ["modelo.ply"]
    try:
        import trimesh

        malha = trimesh.load(str(ply))
        malha.export(str(destino / "modelo.glb"))
        malha.export(str(destino / "modelo.obj"))
        formatos += ["modelo.glb", "modelo.obj"]
        _diz(f"[{_passou():4}s] {len(malha.vertices)} vertices · {len(malha.faces)} faces "
             "· .glb e .obj prontos")
    except Exception as exc:
        _diz(f"aviso: converti so o .ply ({exc})")

    if _previa(ply, destino / "previa.png"):
        formatos.append("previa.png")
        _diz(f"[{_passou():4}s] previa desenhada")

    (destino / "modelo.json").write_text(json.dumps({
        "origem": origem, "pedido": pedido, "foto": foto, "passos": passos,
        "base": FOTO_3D if foto else TEXTO_3D, "segundos": _passou(),
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "licenca_modelo": "Shap-E (OpenAI) — Apache-2.0, uso comercial liberado",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _job["arquivos"] = formatos
    _diz(f"[{_passou():4}s] PRONTO — abra o .glb no Blender ou arraste para o motor de jogo")


def apagar(nome: str) -> dict:
    import shutil

    alvo = raiz() / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "nao encontrado"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "modelos": modelos()}


# ---------------------------------------------------- coletor de referencias
def coletar(termo: str = "", quantidade: int = 6, nome: str = "") -> dict:
    """Baixa modelos 3D do Poly Haven, que e todo CC0 — sem dono, sem atribuicao.

    Serve de referencia: abrir no Blender para ver escala e topologia, ou renderizar
    fotos do objeto para treinar o LoRA de imagem.
    """
    if _coleta["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma coleta rodando"}
    quantidade = max(1, min(int(quantidade or 6), 20))
    pasta = referencias_raiz() / (slug(nome) or slug(termo) or "referencias")
    pasta.mkdir(parents=True, exist_ok=True)
    _coleta.update({"state": "running", "linhas": [], "baixados": 0, "alvo": quantidade,
                    "pasta": str(pasta)})

    def _go() -> None:
        try:
            _diz_coleta("procurando no Poly Haven (tudo CC0)")
            catalogo = json_de(f"{POLYHAVEN}/assets?t=models")
            termos = [t for t in (termo or "").lower().split() if t]
            escolhidos = []
            for ident, dados in catalogo.items():
                texto = " ".join([ident, dados.get("name", ""),
                                  " ".join(dados.get("tags") or []),
                                  " ".join(dados.get("categories") or [])]).lower()
                if not termos or all(t in texto for t in termos):
                    escolhidos.append((ident, dados))
                if len(escolhidos) >= quantidade:
                    break
            if not escolhidos:
                raise RuntimeError(f"nada com \"{termo}\" no Poly Haven; tente outra palavra em ingles")
            _diz_coleta(f"{len(escolhidos)} modelo(s) achado(s)")
            creditos = []
            for ident, dados in escolhidos:
                if _coleta["state"] != "running":
                    break
                try:
                    arquivos = json_de(f"{POLYHAVEN}/files/{ident}")
                    gltf = ((arquivos.get("gltf") or {}).get("1k") or {}).get("gltf")
                    if not gltf:
                        _diz_coleta(f"{ident}: sem gltf de 1k, pulei")
                        continue
                    alvo = pasta / ident
                    alvo.mkdir(parents=True, exist_ok=True)
                    (alvo / f"{ident}.gltf").write_bytes(abrir(gltf["url"], timeout=120))
                    incluidos = gltf.get("include") or {}
                    for relativo, extra in incluidos.items():
                        peca = alvo / relativo
                        peca.parent.mkdir(parents=True, exist_ok=True)
                        peca.write_bytes(abrir(extra["url"], timeout=180))
                    _coleta["baixados"] += 1
                    creditos.append({
                        "id": ident, "nome": dados.get("name", ident),
                        "autores": list((dados.get("authors") or {}).keys()),
                        "licenca": "CC0", "pagina": f"https://polyhaven.com/a/{ident}",
                        "pecas": 1 + len(incluidos),
                    })
                    _diz_coleta(f"{_coleta['baixados']}/{len(escolhidos)} — {dados.get('name', ident)} "
                                f"(CC0, {1 + len(incluidos)} arquivo(s))")
                except Exception as exc:
                    _diz_coleta(f"pulei {ident}: {exc}")
            (pasta / "creditos.json").write_text(json.dumps({
                "termo": termo, "fonte": "Poly Haven", "licenca": "CC0",
                "baixado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "aviso": "CC0: pode usar em projeto comercial sem citar. Citar e educado.",
                "itens": creditos,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            _coleta["state"] = "done"
            _diz_coleta(f"COLETA CONCLUIDA — {_coleta['baixados']} modelo(s) em {pasta}")
        except Exception as exc:
            _coleta["state"] = "error"
            _diz_coleta(f"erro: {exc}")

    threading.Thread(target=_go, name="coletor-3d", daemon=True).start()
    return {"accepted": True, "pasta": str(pasta)}


def parar_coleta() -> dict:
    if _coleta["state"] == "running":
        _coleta["state"] = "done"
        _diz_coleta("parado por voce")
    return status()


def apagar_referencia(nome: str) -> dict:
    import shutil

    alvo = referencias_raiz() / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "nao encontrado"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "referencias": referencias()}


def buscar(termo: str, limite: int = 24) -> dict:
    """Lista o que existe no Poly Haven antes de baixar."""
    try:
        catalogo = json_de(f"{POLYHAVEN}/assets?t=models")
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "itens": []}
    termos = [t for t in (termo or "").lower().split() if t]
    itens = []
    for ident, dados in catalogo.items():
        texto = " ".join([ident, dados.get("name", ""), " ".join(dados.get("tags") or []),
                          " ".join(dados.get("categories") or [])]).lower()
        if not termos or all(t in texto for t in termos):
            itens.append({"id": ident, "nome": dados.get("name", ident),
                          "categorias": dados.get("categories") or [],
                          "previa": f"https://cdn.polyhaven.com/asset_img/thumbs/{ident}.png?width=200",
                          "pagina": f"https://polyhaven.com/a/{urllib.parse.quote(ident)}"})
        if len(itens) >= limite:
            break
    return {"ok": True, "itens": itens, "total": len(itens)}
