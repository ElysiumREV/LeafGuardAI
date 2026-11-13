"""Consulta ao Gemini para orientações de tratamento, sem dependência da interface."""
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lista de reserva, usada só se não for possível consultar os modelos da API.
FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
MAX_AUTO_CANDIDATES = 6
_EXCLUDED = ("tts", "image", "live", "audio", "embedding", "robotics", "computer-use", "vision", "thinking", "learnlm")
_MODEL_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?(?:-(.+))?$")
_RETRYABLE_CODES = (404, 429, 500, 503, 504)
_RETRYABLE_WORDS = ("overloaded", "unavailable", "high demand", "resource_exhausted", "quota", "not found")


@dataclass
class Guidance:
    text: str
    model: str
    failed: list[str] = field(default_factory=list)


class GuidanceError(Exception):
    """Nenhum modelo respondeu. `error` é a última exceção; `failed`, os modelos que falharam."""

    def __init__(self, error, failed):
        super().__init__(str(error))
        self.error = error
        self.failed = failed


def discover_models(client):
    """Consulta a API e devolve os modelos Gemini Flash disponíveis, do mais novo ao mais antigo."""
    found = []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        actions = model.supported_actions or []
        if actions and "generateContent" not in actions:
            continue
        if any(word in name for word in _EXCLUDED):
            continue
        match = _MODEL_RE.match(name)
        if not match:
            continue
        version = tuple(int(x) for x in match.group(1).split("."))
        is_lite = bool(match.group(2))
        suffix = match.group(3) or ""
        # Estáveis (sem sufixo) vêm antes de preview/exp/datados da mesma versão.
        found.append((tuple(-v for v in version), is_lite, suffix != "", name))
    found.sort()
    return [item[3] for item in found]


def _error_code(error):
    return getattr(error, "code", None) or getattr(error, "status_code", None)


def is_retryable(error):
    """Erros em que vale tentar outro modelo (alto uso, cota, indisponível)."""
    low = str(error).lower()
    return _error_code(error) in _RETRYABLE_CODES or any(w in low for w in _RETRYABLE_WORDS)


def describe_error(error):
    raw = str(error)
    low = raw.lower()
    code = _error_code(error)

    if code in (400, 401, 403) or "api key" in low or "api_key" in low or "permission" in low:
        reason = "Chave da API inválida ou sem permissão. Confira se copiou a chave inteira."
    elif code == 404 or "not found" in low:
        reason = "Modelo não encontrado para a sua chave."
    elif code == 429 or "quota" in low or "resource_exhausted" in low:
        reason = "Limite de uso (cota) da API atingido. Tente mais tarde."
    elif any(w in low for w in ("connect", "timeout", "resolve", "network")):
        reason = "Falha de conexão. Verifique sua internet."
    else:
        reason = "Não foi possível obter as orientações da IA."

    return f"{reason}\n\nDetalhe: {type(error).__name__}: {raw[:300]}"


def candidate_models(client, model=None):
    """Um modelo escolhido, ou (automático) os mais novos disponíveis, com a lista de reserva se a consulta falhar."""
    if model:
        return [model]
    try:
        candidates = discover_models(client)[:MAX_AUTO_CANDIDATES]
    except Exception as error:
        logger.warning("listagem de modelos falhou: %s", error)
        candidates = []
    return candidates or list(FALLBACK_MODELS)


def request_guidance(client, prompt, model=None):
    """Pede a resposta ao Gemini. Com `model=None` (automático), troca de modelo se um estiver ocupado.

    Retorna Guidance; levanta GuidanceError se nenhum modelo responder. Erros que não são de
    disponibilidade (ex.: chave inválida) interrompem na hora, sem tentar os outros modelos.
    """
    failed = []
    last_error = None
    for name in candidate_models(client, model):
        try:
            interaction = client.interactions.create(model=name, input=prompt)
        except Exception as error:
            last_error = error
            logger.warning("%s: %s: %s", name, type(error).__name__, error)
            if is_retryable(error):
                failed.append(name)
                continue
            break
        return Guidance(text=interaction.output_text, model=name, failed=failed)
    raise GuidanceError(last_error, failed)
