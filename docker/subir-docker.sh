#!/usr/bin/env bash
# Sobe o daemon do Docker. Serve para o rootless (instalado em ~/.local), que nao tem
# servico de sistema e por isso nao volta sozinho depois de reiniciar a maquina.
set -u
export PATH="$HOME/.local/bin:/usr/sbin:/sbin:$PATH"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock"

if docker info >/dev/null 2>&1; then
  echo "docker ja esta de pe: $(docker --version)"
  exit 0
fi

if command -v systemctl >/dev/null && systemctl --user start docker 2>/dev/null; then
  sleep 3
  docker info >/dev/null 2>&1 && { echo "subiu pelo systemd"; exit 0; }
fi

command -v dockerd-rootless.sh >/dev/null || {
  echo "docker rootless nao encontrado; use o Docker do sistema (sudo systemctl start docker)" >&2
  exit 1
}

# sem iptables nao da para criar a rede bridge; nesse caso o container usa a rede do host
FLAGS=""
command -v iptables >/dev/null || FLAGS="--iptables=false --ip6tables=false --bridge=none"
[ -n "$FLAGS" ] && echo "sem iptables nesta maquina: subindo sem bridge (use docker-compose.hostnet.yml)"

nohup dockerd-rootless.sh $FLAGS > /tmp/dockerd.log 2>&1 &
for _ in $(seq 1 20); do
  sleep 2
  docker info >/dev/null 2>&1 && { echo "docker no ar: $(docker --version)"; exit 0; }
done
echo "nao subiu; veja /tmp/dockerd.log" >&2
tail -5 /tmp/dockerd.log >&2
exit 1
