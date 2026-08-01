# MAG-pal — Discord AI pal bot
# Includes an agent-ready toolset (bash, curl, nmap, build tools, runtimes)
# for future AI tool-calling / MCP features.

FROM python:3.14-slim

# tini: minimal init (PID 1) that reaps zombie processes spawned by the AI
RUN apt-get update && apt-get install -y --no-install-recommends \
    # --- Base / process ---
    tini \
    ca-certificates \
    bash \
    coreutils \
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
    tcpdump \
    nmap \
    # --- File ops ---
    unzip \
    zip \
    xz-utils \
    bzip2 \
    rsync \
    tree \
    diffutils \
    patch \
    lsof \
    # --- Search & text ---
    ripgrep \
    fd-find \
    jq \
    gawk \
    sed \
    # --- Build tools ---
    build-essential \
    make \
    cmake \
    pkg-config \
    gdb \
    # --- Git ---
    git \
    git-lfs \
    gh \
    # --- Runtimes ---
    python3-venv \
    nodejs \
    npm \
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
