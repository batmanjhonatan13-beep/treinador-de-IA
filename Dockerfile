FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

# ssh e tar: a pagina "Preparar ambiente" configura outro servidor a partir daqui
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client tar curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY web /app/web
COPY config.yaml /app/config.yaml
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8765
CMD ["/app/docker/entrypoint.sh"]
