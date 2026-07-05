# ─── FPP Golden Window — Dockerfile ───
# 一键复现论文核心实验（消费级硬件，无需 GPU）
#
# 用法（在仓库根目录）:
#   docker build -t fpp-golden-window -f docker/Dockerfile .
#   或: cd docker && docker compose up
#
# GPU 版本:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 -t fpp-golden-window:gpu -f docker/Dockerfile .

FROM python:3.10-slim

LABEL org.opencontainers.image.title="FPP Golden Window"
LABEL org.opencontainers.image.description="Reproduce 'Loss Is Not Enough: The Golden Window in Neural Network Training' — 4D neural network diagnostics"
LABEL org.opencontainers.image.authors="Song Yue <blackcat>"
LABEL org.opencontainers.image.source="https://github.com/GIS-blackCaat/fpp-golden-window"

# ─── Build arguments ───
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1

# ─── Install system deps ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── Install Python deps ───
# torch installed separately (custom index for CPU/CUDA)
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}
RUN pip install --no-cache-dir \
    transformers>=4.46.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    sentencepiece>=0.1.99 \
    accelerate>=0.20.0 \
    huggingface_hub

# ─── Copy FPP files ───
WORKDIR /app

# Tools
COPY tools/ ./tools/
COPY examples/ ./examples/

# Data, figures, tables
COPY data/ ./data/
COPY figures/ ./figures/
COPY tables/ ./tables/

# Docs
COPY docs/ ./docs/
COPY README.md README_CN.md LICENSE requirements.txt ./

# Entrypoint
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Make tools/examples importable
ENV PYTHONPATH=/app:/app/tools:/app/examples

# ─── HF cache as volume (models persist across runs) ───
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
VOLUME ["/app/.cache"]

# ─── Entrypoint ───
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["demo"]
