#!/usr/bin/env bash
# Sobe o roteador deste pacote. Funciona em qualquer distribuicao: se faltar o ambiente,
# ele chama o instalar.sh (que ja sabe o gerenciador de pacotes do seu sistema) e segue.
#
#   ./subir.sh                      porta 8770, pecas dormem depois de 5 min paradas
#   ./subir.sh --porta 9000         outra porta
#   ./subir.sh --manter 0           nada dorme (mais rapido, come memoria)
#   ./subir.sh --acordar texto      ja deixa o texto ligado ao subir
set -uo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "== primeira vez: preparando o ambiente"
  if [ -x ./instalar.sh ]; then
    ./instalar.sh || { echo "a instalacao falhou; veja as mensagens acima"; exit 1; }
  else
    python3 -m venv .venv && .venv/bin/pip install --upgrade pip
  fi
fi

if [ ! -x .venv/bin/python ]; then
  echo "nao consegui criar o ambiente em .venv — instale o python3 e rode de novo"
  exit 1
fi

if .venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "== placa de video encontrada"
else
  echo "== sem placa de video: texto e som rodam na memoria (devagar); imagem e 3D nao rodam"
fi

exec .venv/bin/python roteador.py "$@"
