import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from src.app_config import get_config_value
except ImportError:
    from app_config import get_config_value

_INIT_LOCK = threading.Lock()
_LOGGING_INITIALIZED = False


def _get_int_env(var_name, default_value):
    """Lê uma variável de ambiente inteira com fallback seguro.

    Parameters
    ----------
    var_name : str
        Nome da variável de ambiente.
    default_value : int
        Valor padrão quando a variável não existe ou é inválida.

    Returns
    -------
    int
        Valor inteiro válido para uso na configuração.
    """
    mapping = {
        "LOG_MAX_BYTES": "logging.max_bytes",
        "LOG_BACKUP_COUNT": "logging.backup_count",
    }
    try:
        return int(get_config_value(mapping.get(var_name, ""), default_value))
    except (TypeError, ValueError):
        return default_value


def _resolve_log_level():
    """Resolve o nível de log a partir do ambiente.

    Returns
    -------
    int
        Nível de log do módulo `logging`.
    """
    level_str = str(get_config_value("logging.level", "INFO")).strip().upper()
    return getattr(logging, level_str, logging.INFO)


def _initialize_logging():
    """Inicializa handlers de logging de forma idempotente.

    Returns
    -------
    None
        Configura o logger raiz para console e arquivo rotativo.
    """
    global _LOGGING_INITIALIZED

    with _INIT_LOCK:
        if _LOGGING_INITIALIZED:
            return

        log_level = _resolve_log_level()
        log_file = str(get_config_value("logging.file", "logs/app.log"))
        log_max_bytes = _get_int_env("LOG_MAX_BYTES", 5_242_880)
        log_backup_count = _get_int_env("LOG_BACKUP_COUNT", 5)

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove apenas handlers gerenciados por este módulo.
        for handler in list(root_logger.handlers):
            if getattr(handler, "_rag_managed", False):
                root_logger.removeHandler(handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        stream_handler._rag_managed = True  # type: ignore[attr-defined]

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler._rag_managed = True  # type: ignore[attr-defined]

        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)
        _LOGGING_INITIALIZED = True


def get_logger(name):
    """Retorna logger configurado para o nome informado.

    Parameters
    ----------
    name : str
        Nome do logger.

    Returns
    -------
    logging.Logger
        Logger pronto para uso.
    """
    _initialize_logging()
    return logging.getLogger(name)
