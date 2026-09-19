"""Classificador de imagem: aprende a separar categorias suas, e prova quanto acerta.

Das coisas que dao para treinar em casa, esta e a unica alem do texto que tem prova de
verdade: parte das imagens fica de fora do treino, e o acerto e medido nelas. Se o numero
sobe no treino e fica baixo na prova, o modelo decorou em vez de aprender — e isso aparece.

Uma pasta por categoria dentro de data/classes/:

    data/classes/pecas/boa/*.jpg
    data/classes/pecas/com-defeito/*.jpg
"""

from __future__ import annotations

import json
import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from agentepc.config import resolve
from agentepc.lotes import slug

EXT = (".jpg", ".jpeg", ".png", ".webp")
MIN_POR_CLASSE = 8

_job: dict = {
    "state": "idle", "linhas": [], "epoca": 0, "epocas": 0, "acerto": 0.0,
    "treino_acerto": 0.0, "conjunto": "", "saida": "", "terminou": False, "consolidado": False,
}


def _diz(msg: str) -> None:
    _job["linhas"].append(msg)
    del _job["linhas"][:-100]


def raiz() -> Path:
    path = resolve("data/classes")
    path.mkdir(parents=True, exist_ok=True)
    return path


def saida_raiz() -> Path:
    path = resolve("data/classificadores")
    path.mkdir(parents=True, exist_ok=True)
    return path


def conjuntos() -> list[dict]:
    saida = []
    for pasta in sorted(raiz().iterdir()) if raiz().exists() else []:
        if not pasta.is_dir():
            continue
        classes = {}
        for sub in sorted(pasta.iterdir()):
            if sub.is_dir():
                classes[sub.name] = len([p for p in sub.glob("*") if p.suffix.lower() in EXT])
        saida.append({"id": pasta.name, "classes": classes, "total": sum(classes.values())})
    return saida


def treinados() -> list[dict]:
    saida = []
    for pasta in sorted(saida_raiz().iterdir()) if saida_raiz().exists() else []:
        meta = pasta / "modelo.json"
        if meta.is_file():
            saida.append({"id": pasta.name, **json.loads(meta.read_text(encoding="utf-8"))})
    return saida


def status() -> dict:
    return {**_job, "conjuntos": conjuntos(), "modelos": treinados()}


def importar(conjunto: str, classe: str, colecao: str) -> dict:
    """Aproveita uma colecao de imagens ja baixada como uma categoria."""
    import shutil

    from agentepc.coletor import raiz as imagens_raiz

    origem = imagens_raiz() / slug(colecao)
    if not origem.is_dir():
        return {"ok": False, "reason": "coleção não encontrada"}
    destino = raiz() / slug(conjunto) / slug(classe)
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in origem.glob("*"):
        if img.suffix.lower() in EXT:
            shutil.copy2(img, destino / img.name)
            n += 1
    return {"ok": True, "copiadas": n, "conjuntos": conjuntos()}


def treinar(conjunto: str, epocas: int = 8, nome: str = "", limiar: float = 0.9) -> dict:
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um treino rodando"}
    from agentepc import capacidades

    ok, motivo = capacidades.liberado("classificador-imagem")
    if not ok:
        return {"accepted": False, "reason": motivo}
    pasta = raiz() / slug(conjunto)
    classes = {p.name: [x for x in p.glob("*") if x.suffix.lower() in EXT]
               for p in sorted(pasta.iterdir()) if p.is_dir()} if pasta.is_dir() else {}
    classes = {k: v for k, v in classes.items() if v}
    if len(classes) < 2:
        return {"accepted": False, "reason": "precisa de pelo menos 2 categorias (2 subpastas)"}
    magras = [k for k, v in classes.items() if len(v) < MIN_POR_CLASSE]
    if magras:
        return {"accepted": False,
                "reason": f"categoria com poucas imagens ({', '.join(magras)}): mínimo {MIN_POR_CLASSE}"}
    destino = saida_raiz() / (slug(nome) or f"{slug(conjunto)}-{datetime.now().strftime('%Y%m%d-%H%M')}")
    destino.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "epoca": 0, "epocas": int(epocas),
                 "acerto": 0.0, "treino_acerto": 0.0, "conjunto": conjunto,
                 "saida": str(destino), "terminou": False, "consolidado": False})

    def _go() -> None:
        try:
            _treino(classes, int(epocas), destino, float(limiar))
            _job["state"] = "done"
            _job["terminou"] = True
        except Exception as exc:
            _job["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="classificador", daemon=True).start()
    return {"accepted": True, "classes": {k: len(v) for k, v in classes.items()}}


def parar() -> dict:
    if _job["state"] == "running":
        _job["state"] = "parando"
        _diz("parando no fim da época")
    return status()


def _treino(classes: dict, epocas: int, destino: Path, limiar: float) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    nomes = sorted(classes)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    _diz(f"{len(nomes)} categoria(s): " + ", ".join(f"{n} ({len(classes[n])})" for n in nomes))

    # separa ANTES de treinar: a prova tem que sair de imagens que o modelo nunca viu
    treino, prova = [], []
    sorteio = random.Random(42)
    for i, nome in enumerate(nomes):
        arquivos = sorted(classes[nome])
        sorteio.shuffle(arquivos)
        corte = max(2, round(len(arquivos) * 0.25))
        prova += [(p, i) for p in arquivos[:corte]]
        treino += [(p, i) for p in arquivos[corte:]]
    _diz(f"{len(treino)} para treinar, {len(prova)} guardadas para a prova")

    prep_treino = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    prep_prova = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class Conjunto(Dataset):
        def __init__(self, itens, prep):
            self.itens, self.prep = itens, prep

        def __len__(self):
            return len(self.itens)

        def __getitem__(self, i):
            caminho, alvo = self.itens[i]
            with Image.open(caminho) as im:
                return self.prep(im.convert("RGB")), alvo

    carga_treino = DataLoader(Conjunto(treino, prep_treino), batch_size=16, shuffle=True)
    carga_prova = DataLoader(Conjunto(prova, prep_prova), batch_size=16)

    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, len(nomes))
    modelo = modelo.to(dispositivo)
    otim = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
    perda_fn = torch.nn.CrossEntropyLoss()
    _diz(f"resnet18 em {dispositivo}, {epocas} época(s)")

    melhor = 0.0
    for epoca in range(1, epocas + 1):
        if _job["state"] != "running":
            break
        modelo.train()
        certos = total = 0
        for x, y in carga_treino:
            x, y = x.to(dispositivo), y.to(dispositivo)
            saida = modelo(x)
            perda = perda_fn(saida, y)
            perda.backward()
            otim.step()
            otim.zero_grad(set_to_none=True)
            certos += (saida.argmax(1) == y).sum().item()
            total += y.numel()
        _job["treino_acerto"] = certos / max(total, 1)

        modelo.eval()
        certos = total = 0
        with torch.no_grad():
            for x, y in carga_prova:
                x, y = x.to(dispositivo), y.to(dispositivo)
                certos += (modelo(x).argmax(1) == y).sum().item()
                total += y.numel()
        acerto = certos / max(total, 1)
        _job["epoca"], _job["acerto"] = epoca, acerto
        _diz(f"época {epoca}/{epocas} — treino {_job['treino_acerto']:.0%} · "
             f"prova {acerto:.0%} ({certos}/{total})")
        if acerto >= melhor:
            melhor = acerto
            torch.save({"pesos": modelo.state_dict(), "classes": nomes}, destino / "modelo.pt")

    _job["consolidado"] = melhor >= limiar
    (destino / "modelo.json").write_text(json.dumps({
        "conjunto": _job["conjunto"], "classes": nomes, "epocas": _job["epoca"],
        "acerto_prova": round(melhor, 4), "acerto_treino": round(_job["treino_acerto"], 4),
        "imagens_treino": len(treino), "imagens_prova": len(prova),
        "limiar": limiar, "consolidado": melhor >= limiar,
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _diz(f"MELHOR ACERTO NA PROVA: {melhor:.0%}"
         + (" — consolidado" if melhor >= limiar else f" — abaixo do limiar de {limiar:.0%}"))
    if _job["treino_acerto"] - melhor > 0.25:
        _diz("acerta muito mais no treino que na prova: decorou. Junte mais imagens variadas.")


def apagar(nome: str) -> dict:
    import shutil

    alvo = saida_raiz() / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "não encontrado"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "modelos": treinados()}
