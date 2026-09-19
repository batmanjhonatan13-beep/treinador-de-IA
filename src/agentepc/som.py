"""Som: juntar audio com licenca clara, classificar som e gerar som.

Sao tres coisas diferentes que costumam ser chamadas de "treino de som", e so uma delas
tem prova automatica:

  1. Classificar   — "isso e um motor batendo ou rodando liso?" Tem resposta certa, entao
                     parte dos audios fica de fora do treino e o acerto e medido neles.
  2. Gerar         — o modelo compoe som novo no estilo do que ouviu. Nao ha resposta
                     certa: quem julga e voce, ouvindo.
  3. Clonar voz    — copiar o timbre de alguem. Fica no catalogo como planejado: o
                     problema ali e licenca, nao GPU (veja capacidades.py).

De onde vem o audio: a Openverse indexa o Freesound e o Wikimedia Commons, e devolve a
licenca de cada faixa junto. Spotify, YouTube e afins ficam de fora de proposito — os
termos de uso deles proibem baixar e treinar, e isso nao muda por a ferramenta ser local.
"""

from __future__ import annotations

import json
import random
import re
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from agentepc.coletor import LICENCAS, UA
from agentepc.config import resolve
from agentepc.lotes import slug

# o que o libsndfile (que vem dentro do soundfile) realmente abre. WebM, m4a e aac ficam
# de fora: o Wikimedia serve muito .webm, e um arquivo desses so aparece na hora do treino,
# com uma mensagem que nao ajuda ninguem ("File does not exist")
EXT = (".mp3", ".wav", ".ogg", ".oga", ".flac", ".opus")
SR = 16000            # taxa do classificador: fala e ruido cabem bem em 16 kHz
JANELA = 3.0          # segundos por amostra
MIN_POR_CLASSE = 8
MAX_SEG = 180         # so os 3 primeiros minutos: uma gravacao de 1 h nao cabe na memoria

_coleta: dict = {"state": "idle", "linhas": [], "baixadas": 0, "alvo": 0, "pasta": "", "termo": ""}
_job: dict = {
    "state": "idle", "linhas": [], "epoca": 0, "epocas": 0, "acerto": 0.0,
    "treino_acerto": 0.0, "conjunto": "", "saida": "", "terminou": False, "consolidado": False,
    "tipo": "classificador",
}
_geracao: dict = {"state": "idle", "linhas": [], "arquivo": "", "pedido": ""}


def _diz_coleta(msg: str) -> None:
    _coleta["linhas"].append(msg)
    del _coleta["linhas"][:-80]


def _diz(msg: str) -> None:
    _job["linhas"].append(msg)
    del _job["linhas"][:-100]


def _diz_ger(msg: str) -> None:
    _geracao["linhas"].append(msg)
    del _geracao["linhas"][:-40]


# ---------------------------------------------------------------- pastas
def raiz() -> Path:
    path = resolve("data/sons")
    path.mkdir(parents=True, exist_ok=True)
    return path


def classes_raiz() -> Path:
    path = resolve("data/som-classes")
    path.mkdir(parents=True, exist_ok=True)
    return path


def modelos_raiz() -> Path:
    path = resolve("data/som-modelos")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gerados_raiz() -> Path:
    path = resolve("data/sons-gerados")
    path.mkdir(parents=True, exist_ok=True)
    return path


def colecoes() -> list[dict]:
    saida = []
    for pasta in sorted(raiz().iterdir()) if raiz().exists() else []:
        if not pasta.is_dir():
            continue
        faixas = [p for p in pasta.glob("*") if p.suffix.lower() in EXT]
        meta = pasta / "creditos.json"
        dados = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        saida.append({
            "id": pasta.name, "faixas": len(faixas),
            "bytes": sum(p.stat().st_size for p in faixas),
            "termo": dados.get("termo", ""), "licenca": dados.get("licenca", ""),
        })
    return saida


def conjuntos() -> list[dict]:
    saida = []
    for pasta in sorted(classes_raiz().iterdir()) if classes_raiz().exists() else []:
        if not pasta.is_dir():
            continue
        classes = {sub.name: len([p for p in sub.glob("*") if p.suffix.lower() in EXT])
                   for sub in sorted(pasta.iterdir()) if sub.is_dir()}
        saida.append({"id": pasta.name, "classes": classes, "total": sum(classes.values())})
    return saida


def treinados() -> list[dict]:
    saida = []
    for pasta in sorted(modelos_raiz().iterdir()) if modelos_raiz().exists() else []:
        meta = pasta / "modelo.json"
        if meta.is_file():
            saida.append({"id": pasta.name, **json.loads(meta.read_text(encoding="utf-8"))})
    return saida


def gerados() -> list[dict]:
    saida = []
    for arq in sorted(gerados_raiz().glob("*.wav"), reverse=True)[:30]:
        meta = arq.with_suffix(".json")
        dados = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        saida.append({"arquivo": arq.name, "bytes": arq.stat().st_size, **dados})
    return saida


def status() -> dict:
    return {
        "coleta": {**_coleta, "colecoes": colecoes()},
        "treino": {**_job, "conjuntos": conjuntos(), "modelos": treinados()},
        "geracao": {**_geracao, "arquivos": gerados()},
    }


# ---------------------------------------------------------------- coletor
def _busca_openverse(termo: str, quantidade: int, licenca: str) -> list[dict]:
    achados, pagina = [], 1
    while len(achados) < quantidade and pagina <= 5:
        params = {"q": termo, "page_size": 50, "page": pagina}
        if LICENCAS.get(licenca):
            params["license"] = LICENCAS[licenca]
        url = "https://api.openverse.org/v1/audio/?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                dados = json.load(r)
        except Exception as exc:
            _diz_coleta(f"openverse: {exc}")
            break
        for item in dados.get("results") or []:
            endereco = item.get("url")
            if not endereco:
                continue
            achados.append({
                "url": endereco, "fonte": item.get("source") or "openverse",
                "autor": item.get("creator") or "?", "licenca": item.get("license") or "?",
                "pagina": item.get("foreign_landing_url") or "", "titulo": item.get("title") or "",
                "duracao_s": round((item.get("duration") or 0) / 1000, 1),
            })
        if not dados.get("results"):
            break
        pagina += 1
    return achados[:quantidade]


def _busca_commons(termo: str, quantidade: int) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:audio {termo}", "gsrnamespace": "6",
        "gsrlimit": str(min(quantidade * 2, 100)),
        "prop": "imageinfo", "iiprop": "url|extmetadata|size",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
            dados = json.load(r)
    except Exception as exc:
        _diz_coleta(f"commons: {exc}")
        return []
    achados = []
    for pag in ((dados.get("query") or {}).get("pages") or {}).values():
        ii = (pag.get("imageinfo") or [{}])[0]
        extra = ii.get("extmetadata") or {}
        achados.append({
            "url": ii.get("url"), "fonte": "wikimedia",
            "autor": re.sub(r"<[^>]+>", "", (extra.get("Artist") or {}).get("value", "?"))[:80],
            "licenca": (extra.get("LicenseShortName") or {}).get("value", "?"),
            "pagina": ii.get("descriptionurl") or "", "titulo": pag.get("title", ""),
            "duracao_s": 0,
        })
    return achados[:quantidade]


def coletar(termo: str, quantidade: int = 20, licenca: str = "livres",
            fontes: str = "openverse,wikimedia", nome: str = "") -> dict:
    """Baixa audio de fontes que declaram a licenca, e guarda a licenca junto."""
    if _coleta["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma coleta rodando"}
    termo = (termo or "").strip()
    if not termo:
        return {"accepted": False, "reason": "diga que som voce procura"}
    quantidade = max(4, min(int(quantidade or 20), 120))
    pasta = raiz() / (slug(nome) or slug(termo) or "colecao")
    pasta.mkdir(parents=True, exist_ok=True)
    _coleta.update({"state": "running", "linhas": [], "baixadas": 0, "alvo": quantidade,
                    "pasta": str(pasta), "termo": termo})

    def _go() -> None:
        try:
            achados: list[dict] = []
            if "openverse" in fontes:
                achados += _busca_openverse(termo, quantidade, licenca)
                _diz_coleta(f"openverse: {len(achados)} candidata(s)")
            if "wikimedia" in fontes and len(achados) < quantidade:
                extras = _busca_commons(termo, quantidade - len(achados))
                achados += extras
                _diz_coleta(f"wikimedia: +{len(extras)} candidata(s)")
            if not achados:
                raise RuntimeError("nada encontrado com essa licenca; troque o termo ou solte a licenca")
            creditos, vistos = [], set()
            for item in achados:
                if _coleta["state"] != "running" or _coleta["baixadas"] >= quantidade:
                    break
                url = item["url"]
                if not url or url in vistos:
                    continue
                vistos.add(url)
                ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
                if ext not in EXT:
                    _diz_coleta(f"pulei {Path(url).name[:40]}: formato {ext or '?'} que o leitor nao abre")
                    continue
                alvo = pasta / f"{_coleta['baixadas'] + 1:03d}{ext}"
                try:
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=90) as r:
                        dados = r.read(60_000_000)
                    if len(dados) < 8_000:
                        continue
                    alvo.write_bytes(dados)
                except Exception as exc:
                    _diz_coleta(f"pulei {url[:50]}: {exc}")
                    continue
                _coleta["baixadas"] += 1
                creditos.append({**item, "arquivo": alvo.name})
                _diz_coleta(f"{_coleta['baixadas']}/{quantidade} — {item['licenca']} — "
                            f"{item['autor'][:28]} — {item['titulo'][:40]}")
            (pasta / "creditos.json").write_text(json.dumps({
                "termo": termo, "licenca": licenca, "fontes": fontes,
                "baixado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "aviso": "Cite autor e licenca de cada faixa se publicar algo treinado com elas.",
                "itens": creditos,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            _coleta["state"] = "done"
            _diz_coleta(f"COLETA CONCLUIDA — {_coleta['baixadas']} faixa(s) em {pasta}")
        except Exception as exc:
            _coleta["state"] = "error"
            _diz_coleta(f"erro: {exc}")

    threading.Thread(target=_go, name="coletor-som", daemon=True).start()
    return {"accepted": True, "pasta": str(pasta)}


def parar_coleta() -> dict:
    if _coleta["state"] == "running":
        _coleta["state"] = "done"
        _diz_coleta("parado por voce")
    return status()


def apagar_colecao(nome: str) -> dict:
    import shutil

    alvo = raiz() / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "colecao nao encontrada"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "colecoes": colecoes()}


def importar(conjunto: str, classe: str, colecao: str) -> dict:
    """Aproveita uma colecao ja baixada como uma categoria do classificador."""
    import shutil

    origem = raiz() / slug(colecao)
    if not origem.is_dir():
        return {"ok": False, "reason": "colecao nao encontrada"}
    destino = classes_raiz() / slug(conjunto) / slug(classe)
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for faixa in origem.glob("*"):
        if faixa.suffix.lower() in EXT:
            shutil.copy2(faixa, destino / faixa.name)
            n += 1
    return {"ok": True, "copiadas": n, "conjuntos": conjuntos()}


# ------------------------------------------------------------------ audio
def carregar(caminho: Path, sr_alvo: int = SR, segundos: float = 0.0):
    """Le qualquer formato comum, vira mono na taxa pedida. Sem ffmpeg: o soundfile
    traz o libsndfile junto, que ja decodifica mp3, ogg, flac e wav."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(caminho)) as arq:
        sr = arq.samplerate
        # ler o arquivo inteiro estoura a memoria num .oga longo; e nao precisa: o treino
        # usa pedacos de poucos segundos
        onda = arq.read(frames=int(sr * MAX_SEG), dtype="float32", always_2d=True)
    onda = onda.mean(axis=1)                      # estereo vira mono
    if sr != sr_alvo:
        from math import gcd
        from scipy.signal import resample_poly
        d = gcd(int(sr), int(sr_alvo))
        onda = resample_poly(onda, sr_alvo // d, sr // d).astype("float32")
    if segundos:
        alvo = int(segundos * sr_alvo)
        onda = onda[:alvo] if len(onda) >= alvo else np.pad(onda, (0, alvo - len(onda)))
    pico = float(abs(onda).max() or 1.0)
    return (onda / pico).astype("float32")


def _filtros_mel(n_mels: int, n_fft: int, sr: int):
    """Banco de filtros triangulares na escala mel — o mesmo que o ouvido faz:
    separa bem o grave e junta o agudo."""
    import numpy as np

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    pontos = mel2hz(np.linspace(hz2mel(0.0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pontos / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype="float32")
    for i in range(n_mels):
        esq, meio, dir_ = bins[i], bins[i + 1], bins[i + 2]
        for j in range(esq, min(meio, fb.shape[1])):
            if meio > esq:
                fb[i, j] = (j - esq) / (meio - esq)
        for j in range(meio, min(dir_, fb.shape[1])):
            if dir_ > meio:
                fb[i, j] = (dir_ - j) / (dir_ - meio)
    return fb


def espectro(onda, sr: int = SR, n_mels: int = 64, n_fft: int = 1024, hop: int = 256):
    """Vira a onda em imagem: tempo na horizontal, altura do som na vertical.
    Com isso o classificador de imagem serve para som sem mudar de arquitetura."""
    import numpy as np
    import torch

    x = torch.from_numpy(np.asarray(onda, dtype="float32"))
    jan = torch.hann_window(n_fft)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop, window=jan, return_complex=True)
    pot = (stft.abs() ** 2)
    fb = torch.from_numpy(_filtros_mel(n_mels, n_fft, sr))
    mel = torch.log1p(fb @ pot)
    return (mel - mel.mean()) / (mel.std() + 1e-5)


# ------------------------------------------------------- classificador de som
def treinar(conjunto: str, epocas: int = 8, nome: str = "", limiar: float = 0.9) -> dict:
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um treino de som rodando"}
    from agentepc import capacidades

    ok, motivo = capacidades.liberado("classificador-som")
    if not ok:
        return {"accepted": False, "reason": motivo}
    pasta = classes_raiz() / slug(conjunto)
    classes = {p.name: [x for x in p.glob("*") if x.suffix.lower() in EXT]
               for p in sorted(pasta.iterdir()) if p.is_dir()} if pasta.is_dir() else {}
    classes = {k: v for k, v in classes.items() if v}
    if len(classes) < 2:
        return {"accepted": False, "reason": "precisa de pelo menos 2 categorias (2 subpastas)"}
    magras = [k for k, v in classes.items() if len(v) < MIN_POR_CLASSE]
    if magras:
        return {"accepted": False,
                "reason": f"categoria com poucas faixas ({', '.join(magras)}): minimo {MIN_POR_CLASSE}"}
    destino = modelos_raiz() / (slug(nome) or f"{slug(conjunto)}-{datetime.now().strftime('%Y%m%d-%H%M')}")
    destino.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "epoca": 0, "epocas": int(epocas),
                 "acerto": 0.0, "treino_acerto": 0.0, "conjunto": conjunto,
                 "saida": str(destino), "terminou": False, "consolidado": False,
                 "tipo": "classificador"})

    def _go() -> None:
        try:
            _treino_classificador(classes, int(epocas), destino, float(limiar))
            _job["state"] = "done"
            _job["terminou"] = True
        except Exception as exc:
            _job["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="som-classificador", daemon=True).start()
    return {"accepted": True, "classes": {k: len(v) for k, v in classes.items()}}


def parar() -> dict:
    if _job["state"] == "running":
        _job["state"] = "parando"
        _diz("parando no fim da epoca")
    return status()


def _fatias(caminho: Path, maximo: int = 6):
    """Um audio longo vira varias amostras de 3 s. Trecho quase mudo e descartado:
    silencio nao ensina nada e ainda estraga a conta do acerto."""
    import numpy as np

    onda = carregar(caminho)
    passo = int(JANELA * SR)
    saida = []
    for ini in range(0, max(len(onda) - passo + 1, 1), passo):
        pedaco = onda[ini:ini + passo]
        if len(pedaco) < passo:
            pedaco = np.pad(pedaco, (0, passo - len(pedaco)))
        if float(np.abs(pedaco).mean()) < 0.005:
            continue
        saida.append(pedaco)
        if len(saida) >= maximo:
            break
    return saida


def _treino_classificador(classes: dict, epocas: int, destino: Path, limiar: float) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    nomes = sorted(classes)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    _diz(f"{len(nomes)} categoria(s): " + ", ".join(f"{n} ({len(classes[n])})" for n in nomes))

    # a separacao e por ARQUIVO, nunca por pedaco: dois pedacos da mesma faixa sao quase
    # iguais, e um de cada lado daria um acerto alto e mentiroso
    sorteio = random.Random(42)
    treino, prova = [], []
    def _pedacos(arq: Path):
        try:
            return _fatias(arq)
        except Exception as exc:
            _diz(f"pulei {arq.name}: {exc}")
            return []

    for i, nome in enumerate(nomes):
        arquivos = sorted(classes[nome])
        sorteio.shuffle(arquivos)
        corte = max(2, round(len(arquivos) * 0.25))
        for arq in arquivos[:corte]:
            prova += [(p, i) for p in _pedacos(arq)]
        for arq in arquivos[corte:]:
            treino += [(p, i) for p in _pedacos(arq)]
    if not treino or not prova:
        raise RuntimeError("audio demais em silencio ou curto demais; junte faixas maiores")
    _diz(f"{len(treino)} pedaco(s) para treinar, {len(prova)} de faixas nunca ouvidas para a prova")

    class Conjunto(Dataset):
        def __init__(self, itens, aumenta):
            self.itens, self.aumenta = itens, aumenta

        def __len__(self):
            return len(self.itens)

        def __getitem__(self, i):
            onda, alvo = self.itens[i]
            if self.aumenta:
                onda = onda * random.uniform(0.7, 1.3)
            mel = espectro(onda).unsqueeze(0)
            mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(224, 224),
                                                  mode="bilinear", align_corners=False)[0]
            return mel.repeat(3, 1, 1), alvo

    carga_treino = DataLoader(Conjunto(treino, True), batch_size=16, shuffle=True)
    carga_prova = DataLoader(Conjunto(prova, False), batch_size=16)

    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, len(nomes))
    modelo = modelo.to(dispositivo)
    otim = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
    perda_fn = torch.nn.CrossEntropyLoss()
    _diz(f"resnet18 sobre espectrograma em {dispositivo}, {epocas} epoca(s)")

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
        _diz(f"epoca {epoca}/{epocas} — treino {_job['treino_acerto']:.0%} · "
             f"prova {acerto:.0%} ({certos}/{total})")
        if acerto >= melhor:
            melhor = acerto
            torch.save({"pesos": modelo.state_dict(), "classes": nomes,
                        "sr": SR, "janela": JANELA}, destino / "modelo.pt")

    _job["consolidado"] = melhor >= limiar
    (destino / "modelo.json").write_text(json.dumps({
        "conjunto": _job["conjunto"], "classes": nomes, "epocas": _job["epoca"],
        "acerto_prova": round(melhor, 4), "acerto_treino": round(_job["treino_acerto"], 4),
        "pedacos_treino": len(treino), "pedacos_prova": len(prova),
        "limiar": limiar, "consolidado": melhor >= limiar, "tipo": "classificador-som",
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _diz(f"MELHOR ACERTO NA PROVA: {melhor:.0%}"
         + (" — consolidado" if melhor >= limiar else f" — abaixo do limiar de {limiar:.0%}"))
    if _job["treino_acerto"] - melhor > 0.25:
        _diz("acerta muito mais no treino que na prova: decorou. Junte faixas mais variadas.")


def ouvir(modelo_id: str, arquivo: str) -> dict:
    """Poe o classificador para dizer o que ouviu num audio solto."""
    import torch
    from torchvision import models

    pasta = modelos_raiz() / slug(modelo_id)
    if not (pasta / "modelo.pt").is_file():
        return {"ok": False, "reason": "classificador nao encontrado"}
    caminho = Path(arquivo)
    if not caminho.is_file():
        # a pagina manda o caminho relativo a data/ (ex.: "sons/chuva/001.mp3")
        for base in (resolve("data"), raiz()):
            tentativa = base / arquivo
            if tentativa.is_file():
                caminho = tentativa
                break
    if not caminho.is_file():
        return {"ok": False, "reason": "audio nao encontrado"}
    dados = torch.load(pasta / "modelo.pt", map_location="cpu", weights_only=False)
    rede = models.resnet18()
    rede.fc = torch.nn.Linear(rede.fc.in_features, len(dados["classes"]))
    rede.load_state_dict(dados["pesos"])
    rede.eval()
    votos = torch.zeros(len(dados["classes"]))
    pedacos = _fatias(caminho, maximo=10)
    if not pedacos:
        return {"ok": False, "reason": "audio curto demais ou em silencio"}
    with torch.no_grad():
        for onda in pedacos:
            mel = espectro(onda).unsqueeze(0).unsqueeze(0)
            mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode="bilinear",
                                                  align_corners=False)[0].repeat(3, 1, 1)
            votos += torch.softmax(rede(mel.unsqueeze(0))[0], dim=0)
    votos = votos / votos.sum()
    ordem = votos.argsort(descending=True)
    return {"ok": True, "resposta": dados["classes"][int(ordem[0])],
            "certeza": round(float(votos[ordem[0]]), 3),
            "todas": {dados["classes"][int(i)]: round(float(votos[i]), 3) for i in ordem}}


def apagar(nome: str) -> dict:
    import shutil

    alvo = modelos_raiz() / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "nao encontrado"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "modelos": treinados()}


# ------------------------------------------------------------------ gerar som
MUSICGEN = "facebook/musicgen-small"


def gerar(pedido: str, segundos: int = 8, estilo: str = "") -> dict:
    """Escreve som novo a partir de uma frase. Com `estilo`, usa um LoRA seu por cima."""
    if _geracao["state"] == "running":
        return {"accepted": False, "reason": "ja tem uma geracao rodando"}
    from agentepc import capacidades

    ok, motivo = capacidades.liberado("som-gerar")
    if not ok:
        return {"accepted": False, "reason": motivo}
    pedido = (pedido or "").strip()
    if not pedido:
        return {"accepted": False, "reason": "descreva o som que voce quer"}
    segundos = max(2, min(int(segundos or 8), 30))
    _geracao.update({"state": "running", "linhas": [], "arquivo": "", "pedido": pedido})

    def _go() -> None:
        try:
            import soundfile as sf
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
            _diz_ger(f"carregando {MUSICGEN} em {dispositivo}")
            proc = AutoProcessor.from_pretrained(MUSICGEN)
            modelo = MusicgenForConditionalGeneration.from_pretrained(
                MUSICGEN, dtype=torch.float16 if dispositivo == "cuda" else torch.float32)
            if estilo:
                pasta = resolve("data/som-lora") / slug(estilo)
                if (pasta / "adapter_config.json").is_file():
                    from peft import PeftModel
                    modelo.decoder = PeftModel.from_pretrained(modelo.decoder, str(pasta))
                    _diz_ger(f"com o seu treino: {estilo}")
                else:
                    _diz_ger(f"aviso: nao achei o treino {estilo}; gerando so com o modelo base")
            modelo = modelo.to(dispositivo)
            entrada = proc(text=[pedido], padding=True, return_tensors="pt").to(dispositivo)
            # o MusicGen conta em fichas: ~50 por segundo de audio
            fichas = int(segundos * 50)
            _diz_ger(f"gerando {segundos}s ({fichas} fichas)")
            with torch.no_grad():
                onda = modelo.generate(**entrada, do_sample=True, guidance_scale=3.0,
                                       max_new_tokens=fichas)
            sr = modelo.config.audio_encoder.sampling_rate
            nome = datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav"
            destino = gerados_raiz() / nome
            sf.write(str(destino), onda[0, 0].float().cpu().numpy(), sr)
            destino.with_suffix(".json").write_text(json.dumps({
                "pedido": pedido, "estilo": estilo, "segundos": segundos, "sr": sr,
                "modelo": MUSICGEN,
                "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            _geracao.update({"state": "done", "arquivo": nome})
            _diz_ger(f"PRONTO — {nome}")
        except Exception as exc:
            _geracao["state"] = "error"
            _diz_ger(f"erro: {exc}")

    threading.Thread(target=_go, name="gerar-som", daemon=True).start()
    return {"accepted": True}


def geracao_status() -> dict:
    return {**_geracao, "arquivos": gerados()}


def apagar_gerado(nome: str) -> dict:
    alvo = gerados_raiz() / Path(nome).name
    if alvo.is_file():
        alvo.unlink()
        alvo.with_suffix(".json").unlink(missing_ok=True)
        return {"ok": True, "arquivos": gerados()}
    return {"ok": False, "reason": "nao encontrado"}


# --------------------------------------------------- treinar a geracao (LoRA)
def treinar_geracao(colecao: str, passos: int = 200, nome: str = "", gatilho: str = "") -> dict:
    """Ensina um jeito de soar ao MusicGen: as faixas viram fichas de audio e o LoRA
    aprende a continuar nesse estilo. Nao ha prova automatica — quem julga e o ouvido."""
    if _job["state"] == "running":
        return {"accepted": False, "reason": "ja tem um treino de som rodando"}
    from agentepc import capacidades

    ok, motivo = capacidades.liberado("som-estilo")
    if not ok:
        return {"accepted": False, "reason": motivo}
    pasta = raiz() / slug(colecao)
    faixas = sorted(p for p in pasta.glob("*") if p.suffix.lower() in EXT) if pasta.is_dir() else []
    if len(faixas) < 4:
        return {"accepted": False, "reason": "precisa de pelo menos 4 faixas nessa colecao"}
    creditos = pasta / "creditos.json"
    termo = (json.loads(creditos.read_text(encoding="utf-8")).get("termo")
             if creditos.is_file() else "") or colecao
    gatilho = (gatilho or termo).strip()
    destino = resolve("data/som-lora") / (slug(nome) or slug(colecao))
    destino.mkdir(parents=True, exist_ok=True)
    _job.update({"state": "running", "linhas": [], "epoca": 0, "epocas": int(passos),
                 "acerto": 0.0, "treino_acerto": 0.0, "conjunto": colecao,
                 "saida": str(destino), "terminou": False, "consolidado": False,
                 "tipo": "estilo"})

    def _go() -> None:
        try:
            _treino_geracao(faixas, int(passos), destino, gatilho)
            _job["state"] = "done"
            _job["terminou"] = True
        except Exception as exc:
            _job["state"] = "error"
            _diz(f"erro: {exc}")

    threading.Thread(target=_go, name="som-estilo", daemon=True).start()
    return {"accepted": True, "faixas": len(faixas), "gatilho": gatilho}


def _treino_geracao(faixas: list, passos: int, destino: Path, gatilho: str) -> None:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    _diz(f"carregando {MUSICGEN} em {dispositivo}")
    proc = AutoProcessor.from_pretrained(MUSICGEN)
    modelo = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN, dtype=torch.float32)
    modelo = modelo.to(dispositivo)
    sr = modelo.config.audio_encoder.sampling_rate

    _diz(f"lendo {len(faixas)} faixa(s) a {sr} Hz, {int(JANELA * 2)}s de cada")
    trechos = []
    for arq in faixas:
        try:
            onda = carregar(arq, sr_alvo=sr, segundos=JANELA * 2)
        except Exception as exc:
            _diz(f"pulei {arq.name}: {exc}")
            continue
        if float(np.abs(onda).mean()) < 0.005:
            _diz(f"pulei {arq.name}: quase em silencio")
            continue
        trechos.append(onda)
    if len(trechos) < 2:
        raise RuntimeError("nao consegui ler audio suficiente dessa colecao")

    # as fichas de audio sao o alvo do treino: o EnCodec transforma onda em numero
    _diz("convertendo audio em fichas (EnCodec)")
    alvos = []
    with torch.no_grad():
        for onda in trechos:
            entrada = torch.from_numpy(onda)[None, None, :].to(dispositivo)
            codigos = modelo.audio_encoder.encode(entrada)[0]        # (1, B, K, T)
            alvos.append(codigos[0, 0].cpu())                        # (K, T)

    # congela TUDO antes do LoRA: sem isso o otimizador leva junto o codificador de texto
    # inteiro (177M de pesos), o que nao e treino de estilo, e destroi o modelo
    for peso in modelo.parameters():
        peso.requires_grad_(False)
    alvo_modulos = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    modelo.decoder = get_peft_model(
        modelo.decoder,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                   target_modules=alvo_modulos),
    )
    # o MusicGen guarda esses dois so no generation_config; sem eles, passar rotulos quebra
    # em "Make sure to set the decoder_start_token_id"
    # o shift dos rotulos le config.decoder.decoder_start_token_id, que vem vazio; o valor
    # certo (2048) so esta no generation_config e no bos do decodificador
    if getattr(modelo.config.decoder, "decoder_start_token_id", None) is None:
        modelo.config.decoder.decoder_start_token_id = (
            modelo.generation_config.decoder_start_token_id or modelo.config.decoder.bos_token_id)
    if modelo.config.decoder.pad_token_id is None:
        modelo.config.decoder.pad_token_id = modelo.generation_config.pad_token_id
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    _diz(f"{treinaveis / 1e6:.1f}M pesos treinaveis (rank 16) · gatilho: \"{gatilho}\"")

    otim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=1e-4)
    texto = proc(text=[gatilho], padding=True, return_tensors="pt").to(dispositivo)
    modelo.train()
    sorteio = random.Random(42)
    inicio = datetime.now()
    for passo in range(1, passos + 1):
        if _job["state"] != "running":
            _diz("parado por voce")
            break
        codigos = sorteio.choice(alvos).to(dispositivo)
        saida = modelo(
            input_ids=texto["input_ids"], attention_mask=texto.get("attention_mask"),
            labels=codigos.T.unsqueeze(0),        # (1, tempo, livros)
        )
        saida.loss.backward()
        otim.step()
        otim.zero_grad(set_to_none=True)
        _job["epoca"] = passo
        if passo % 20 == 0 or passo == 1:
            _diz(f"passo {passo}/{passos} — perda {float(saida.loss):.3f} "
                 f"({(datetime.now() - inicio).seconds}s)")

    modelo.decoder.save_pretrained(str(destino))
    (destino / "treino.json").write_text(json.dumps({
        "colecao": _job["conjunto"], "gatilho": gatilho, "passos": _job["epoca"],
        "faixas": len(trechos), "base": MUSICGEN, "tipo": "som-estilo",
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _diz(f"pesos salvos em {destino}")
    _diz("TREINO CONCLUIDO — gere um som com esse estilo e diga se ficou parecido")


def estilos() -> list[dict]:
    """LoRAs de som ja treinados."""
    pasta = resolve("data/som-lora")
    saida = []
    for item in sorted(pasta.iterdir()) if pasta.exists() else []:
        meta = item / "treino.json"
        if meta.is_file():
            saida.append({"id": item.name, **json.loads(meta.read_text(encoding="utf-8"))})
    return saida


def apagar_estilo(nome: str) -> dict:
    import shutil

    alvo = resolve("data/som-lora") / slug(nome)
    if not alvo.is_dir():
        return {"ok": False, "reason": "nao encontrado"}
    shutil.rmtree(alvo, ignore_errors=True)
    return {"ok": True, "estilos": estilos()}
