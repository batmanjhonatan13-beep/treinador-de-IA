#!/usr/bin/env bash
# Acha um Docker que funcione e deixa o endereco dele em DOCKER_HOST.
#
# Nesta maquina pode haver dois: o Docker Desktop (socket /var/run/docker.sock, do grupo
# "docker") e o rootless instalado na pasta do usuario (/run/user/<id>/docker.sock).
# O script tenta o que ja responde; se o do sistema recusar por permissao, ele diz como
# resolver e usa o rootless enquanto isso.
set -u
export PATH="$HOME/.local/bin:/usr/sbin:/sbin:$PATH"
RUNTIME="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
ROOTLESS="unix://$RUNTIME/docker.sock"
SISTEMA="unix:///var/run/docker.sock"

testa() { DOCKER_HOST="$1" docker info >/dev/null 2>&1; }
erro_de() { DOCKER_HOST="$1" docker info 2>&1 | head -2; }

command -v docker >/dev/null || {
  echo "docker nao esta instalado. No Windows: instale o Docker Desktop e ligue a integracao com o WSL." >&2
  exit 1
}

# 1) o que ja estiver de pe
for alvo in "$SISTEMA" "$ROOTLESS"; do
  if testa "$alvo"; then
    echo "docker no ar em $alvo ($(DOCKER_HOST=$alvo docker version --format '{{.Server.Version}}' 2>/dev/null))"
    echo "export DOCKER_HOST=$alvo"
    exit 0
  fi
done

# 2) o do sistema existe mas recusa: quase sempre e o usuario fora do grupo "docker"
if [ -S /var/run/docker.sock ] && erro_de "$SISTEMA" | grep -qi "permission denied"; then
  echo "O Docker do sistema esta rodando, mas seu usuario nao tem permissao no socket." >&2
  echo "Resolva uma vez com:" >&2
  echo "    sudo usermod -aG docker \$USER" >&2
  echo "  e abra um terminal novo (ou, no Windows: wsl --shutdown)." >&2
fi

# 3) sem nada de pe, sobe o rootless se ele existir
if command -v dockerd-rootless.sh >/dev/null; then
  FLAGS=""
  command -v iptables >/dev/null || FLAGS="--iptables=false --ip6tables=false --bridge=none"
  [ -n "$FLAGS" ] && echo "sem iptables: subindo sem bridge (use docker-compose.hostnet.yml)" >&2
  nohup dockerd-rootless.sh $FLAGS > /tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 20); do
    sleep 2
    if testa "$ROOTLESS"; then
      echo "docker rootless no ar"
      echo "export DOCKER_HOST=$ROOTLESS"
      exit 0
    fi
  done
  echo "o rootless nao subiu; veja /tmp/dockerd.log" >&2
fi

echo "nenhum docker disponivel" >&2
exit 1
