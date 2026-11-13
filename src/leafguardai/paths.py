"""Onde o app guarda o modelo treinado."""
import os
import sys
from pathlib import Path

MODELS_DIR_ENV = "LEAFGUARDAI_MODELS_DIR"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _is_source_checkout():
    """True quando o código roda direto do repositório (e não de um pacote instalado)."""
    return (_REPO_ROOT / "pyproject.toml").is_file() and (_REPO_ROOT / "src" / "leafguardai").is_dir()


def user_data_dir():
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / "LeafGuardAI"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LeafGuardAI"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "leafguardai"


def models_dir():
    """Pasta do modelo, em ordem de prioridade:

    1. a variável de ambiente LEAFGUARDAI_MODELS_DIR;
    2. `models/` na raiz do repositório, quando rodando do código-fonte;
    3. a pasta de dados do usuário, quando instalado como pacote.
    """
    override = os.environ.get(MODELS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if _is_source_checkout():
        return _REPO_ROOT / "models"
    return user_data_dir() / "models"
