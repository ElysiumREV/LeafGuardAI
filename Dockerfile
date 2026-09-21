FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# A imagem-base já contém PyTorch e torchvision compilados para ROCm. O pip
# preserva essas versões porque elas satisfazem as restrições do projeto.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir -e .

CMD ["python", "scripts/train.py"]
