#!/usr/bin/env sh
# Instala o Docker (se faltar) e sobe a pagina do agente-pc.
# Funciona em Linux (apt, dnf, yum, zypper, pacman, apk) e macOS. No Windows use instalar.ps1.
set -e
cd "$(dirname "$0")"

SUDO=""
[ "$(id -u)" = 0 ] || { command -v sudo >/dev/null && SUDO="sudo"; }
PORTA="${PORTA:-8765}"

diga() { printf '\n== %s\n' "$1"; }

diga "verificando o Docker"
if command -v docker >/dev/null 2>&1; then
  docker --version
  # parado, ou sem permissao no socket: o script acha um que funcione e diz o endereco
  if ! docker info >/dev/null 2>&1; then
    SAIDA="$(sh "$(dirname "$0")/docker/subir-docker.sh" 2>&1)" || true
    echo "$SAIDA" | grep -v '^export ' || true
    EXPORTA="$(echo "$SAIDA" | grep '^export DOCKER_HOST=' | tail -1)"
    [ -n "$EXPORTA" ] && eval "$EXPORTA" && export DOCKER_HOST
  fi
else
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null || { echo "instale o Homebrew ou o Docker Desktop e rode de novo"; exit 1; }
      brew install --cask docker
      echo "abra o Docker Desktop uma vez e rode este script de novo"; exit 1 ;;
    *)
      echo "instalando pelo instalador oficial"
      curl -fsSL https://get.docker.com | $SUDO sh
      $SUDO usermod -aG docker "$(id -un)" 2>/dev/null || true ;;
  esac
fi

# docker compose v2 (plugin) ou o docker-compose antigo
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "docker compose nao encontrado"; exit 1
fi

# daemon sem iptables (rootless em WSL) nao cria bridge: nesse caso o container usa a rede
# do host. Erro de permissao e outra coisa — nao pode ser confundido com falta de bridge.
OVERLAY=""
PROBE="$(docker network create --driver bridge _agentepc_probe 2>&1)" && ACHOU=1 || ACHOU=0
if [ "$ACHOU" = "1" ]; then
  docker network rm _agentepc_probe >/dev/null 2>&1 || true
elif echo "$PROBE" | grep -qi "permission denied"; then
  echo "sem permissao no socket do Docker. Rode: sudo usermod -aG docker \$USER" >&2
  echo "e abra um terminal novo (no Windows: wsl --shutdown)." >&2
  exit 1
else
  echo "sem rede bridge: usando a rede do host"
  OVERLAY="-f docker-compose.yml -f docker-compose.hostnet.yml"
fi

diga "subindo a pagina"
$COMPOSE $OVERLAY up -d --build

echo
echo "Pronto: http://127.0.0.1:${PORTA}"
echo "Na pagina, use 'Preparar ambiente' para instalar o modelo e o treino aqui ou em outro servidor."
echo "Parar: $COMPOSE down"
