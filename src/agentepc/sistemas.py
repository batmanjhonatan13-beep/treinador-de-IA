"""Receitas de instalacao por sistema.

Cada distribuicao tem nome de pacote e jeito proprio. O mesmo catalogo serve para
preparar um servidor (pagina Ambiente) e para gerar o instalador que vai dentro do
pacote exportado.
"""

from __future__ import annotations

# familia: como instalar pacote. pacotes: o que o treino precisa alem do Python.
SISTEMAS: list[dict] = [
    {
        "id": "ubuntu",
        "nome": "Ubuntu / Debian",
        "familia": "apt",
        "detecta": ("ubuntu", "debian", "linuxmint", "pop", "raspbian"),
        "pacotes": "python3 python3-venv python3-pip build-essential curl git ca-certificates",
        "extra": "",
    },
    {
        "id": "amzn2023",
        "nome": "Amazon Linux 2023",
        "familia": "dnf",
        "detecta": ("amzn:2023",),
        "pacotes": "python3 python3-pip python3-devel gcc gcc-c++ make curl git tar",
        "extra": "",
    },
    {
        "id": "amzn2",
        "nome": "Amazon Linux 2",
        "familia": "yum",
        "detecta": ("amzn:2",),
        # o Python 3.7 de fabrica nao serve para o torch; o 3.8 vem pelos extras
        "pacotes": "gcc gcc-c++ make curl git tar",
        "extra": (
            'if ! python3.8 -V >/dev/null 2>&1; then\n'
            '  $SUDO amazon-linux-extras install -y python3.8 || true\n'
            'fi\n'
            'command -v python3.8 >/dev/null && PY=python3.8 || PY=python3\n'
            'echo "python do treino: $($PY -V)"\n'
        ),
    },
    {
        "id": "rhel",
        "nome": "RHEL / Rocky / Alma / Fedora",
        "familia": "dnf",
        "detecta": ("rhel", "rocky", "almalinux", "centos", "fedora"),
        "pacotes": "python3 python3-pip python3-devel gcc gcc-c++ make curl git tar",
        "extra": "",
    },
    {
        "id": "suse",
        "nome": "SUSE / openSUSE",
        "familia": "zypper",
        "detecta": ("sles", "opensuse", "opensuse-leap", "opensuse-tumbleweed", "suse"),
        "pacotes": "python3 python3-pip python3-devel gcc gcc-c++ make curl git tar",
        "extra": "",
    },
    {
        "id": "arch",
        "nome": "Arch / Manjaro",
        "familia": "pacman",
        "detecta": ("arch", "manjaro", "endeavouros"),
        "pacotes": "python python-pip base-devel curl git",
        "extra": "",
    },
    {
        "id": "alpine",
        "nome": "Alpine",
        "familia": "apk",
        "detecta": ("alpine",),
        "pacotes": "python3 py3-pip build-base curl git",
        "extra": "",
    },
    {
        "id": "macos",
        "nome": "macOS",
        "familia": "brew",
        "detecta": ("darwin",),
        "pacotes": "python git",
        "extra": "",
    },
    {
        "id": "windows",
        "nome": "Windows",
        "familia": "windows",
        "detecta": ("windows",),
        "pacotes": "",
        "extra": "",
    },
]

INSTALA = {
    "apt": '$SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y',
    "dnf": "$SUDO dnf install -y",
    "yum": "$SUDO yum install -y",
    "zypper": "$SUDO zypper --non-interactive install",
    "pacman": "$SUDO pacman -Sy --noconfirm",
    "apk": "$SUDO apk add --no-cache",
    "brew": "brew install",
}


def por_id(sistema_id: str) -> dict | None:
    return next((s for s in SISTEMAS if s["id"] == sistema_id), None)


def detecta(info: dict) -> str:
    """Escolhe o sistema a partir do que o sniff achou (distro, versao, os)."""
    if info.get("windows") or str(info.get("os", "")).lower() == "windows":
        return "windows"
    if str(info.get("os", "")).lower() == "darwin":
        return "macos"
    distro = str(info.get("distro", "")).lower()
    versao = str(info.get("versao", "")).split(".")[0]
    for s in SISTEMAS:
        for marca in s["detecta"]:
            if ":" in marca:
                d, v = marca.split(":")
                if distro == d and versao == v:
                    return s["id"]
            elif distro == marca:
                return s["id"]
    # sem casar pelo nome, vale o gerenciador de pacotes encontrado
    pkg = info.get("pkg", "")
    for s in SISTEMAS:
        if INSTALA.get(s["familia"], "").split()[-2:-1] == [pkg] or s["familia"].startswith(str(pkg)):
            return s["id"]
    return "ubuntu"


def bloco_basico(sistema_id: str) -> str:
    """Linhas de shell que instalam o basico naquele sistema."""
    s = por_id(sistema_id) or por_id("ubuntu")
    if s["familia"] == "windows":
        return ""
    linhas = [f'echo "== basico ({s["nome"]})"']
    if s["pacotes"]:
        linhas.append(f'{INSTALA[s["familia"]]} {s["pacotes"]}')
    if s["extra"]:
        linhas.append(s["extra"].rstrip())
    return "\n".join(linhas) + "\n"


def nomes() -> list[dict]:
    return [{"id": s["id"], "nome": s["nome"]} for s in SISTEMAS]
