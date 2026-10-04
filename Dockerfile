FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala dependências de sistema necessárias para o OpenCV
# Trocado libgl1-mesa-glx por libgl1 para compatibilidade com Ubuntu 26.04
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# A imagem-base já contém PyTorch e torchvision compilados para ROCm. O pip
# preserva essas versões porque elas satisfazem as restrições do projeto.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir -e .

CMD ["python", "scripts/train.py"]
