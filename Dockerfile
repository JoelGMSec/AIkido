FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    nodriver \
    neotermcolor \
    hypercorn \
    quart \
    requests

WORKDIR /AIkido
COPY . /AIkido
ENTRYPOINT ["python", "AIkido.py"]
