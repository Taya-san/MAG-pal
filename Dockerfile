# MAG-pal — Discord AI pal bot
# Agent-ready toolset for future AI tool-calling / MCP features.

FROM python:3.14-slim

# tini: minimal init (PID 1) that reaps zombie processes spawned by the AI
RUN apt-get update && apt-get install -y --no-install-recommends \
    # --- Base / process ---
    tini \
    ca-certificates \
    bash \
    procps \
    file \
    # --- Network ---
    curl \
    wget \
    socat \
    iproute2 \
    dnsutils \
    iputils-ping \
    openssl \
    netcat-openbsd \
    traceroute \
    nmap \
    # --- File ops ---
    unzip \
    zip \
    xz-utils \
    rsync \
    tree \
    diffutils \
    patch \
    lsof \
    # --- Search & text ---
    ripgrep \
    jq \
    gawk \
    # --- Build tools (needed for pip sdist fallbacks) ---
    build-essential \
    pkg-config \
    # --- Git ---
    git \
    # --- Runtimes ---
    python3-venv \
    # --- Diagnostics ---
    strace \
    htop \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root user (fixed UID for consistency)
RUN groupadd -r app && useradd --no-log-init -r -g app -u 10001 app \
    && mkdir -p /app/data && chown -R app:app /app

USER app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "main.py"]
