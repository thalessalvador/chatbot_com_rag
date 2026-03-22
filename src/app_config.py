"""Utilitarios de configuracao centralizada do projeto.

Este modulo carrega o arquivo `config.yaml` da raiz do projeto e fornece
acesso tipado as configuracoes para todos os componentes do sistema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None


def _project_root() -> Path:
    """Retorna o diretorio raiz do projeto.

    Parameters
    ----------
    None
        Funcao sem parametros.

    Returns
    -------
    Path
        Caminho absoluto da raiz do projeto.
    """
    return Path(__file__).resolve().parent.parent


def _default_config() -> dict[str, Any]:
    """Constroi configuracao padrao quando `config.yaml` nao existe.

    Parameters
    ----------
    None
        Funcao sem parametros.

    Returns
    -------
    dict[str, Any]
        Dicionario com valores padrao seguros.
    """
    return {
        "llm": {
            "provider": "ollama",
            "google_model": "gemini-2.5-flash",
            "google_temperature": 0.0,
            "ollama_model": "ministral-3:8b",
            "ollama_base_url": "http://localhost:11434",
            "ollama_temperature": 0.0,
            "ollama_num_ctx": 3072,
            "ollama_num_predict": 2048,
            "ollama_num_gpu": 9999,
            "ollama_num_thread": 0,
            "ollama_keep_alive": "5m",
        },
        "embeddings": {
            "model": "all-MiniLM-L6-v2",
            "device": "cuda",
            "batch_size": 64,
        },
        "retrieval": {
            "top_k": 5,
            "rrf_k": 60,
            "chroma_add_batch_size": 512,
            "reranking": {
                "enabled": False,
                "model": "BAAI/bge-reranker-v2-m3",
                "device": "cuda",
                "candidate_pool_size": 20,
            },
        },
        "evaluation": {
            "golden_file": "Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx",
            "recall_k": 5,
        },
        "scraping": {
            "base_url": "https://appasp.economia.go.gov.br/pareceres/",
            "max_docs": 9999,
            "timeout_seconds": 30,
            "user_agent": "chatbot-rag-scraper/1.0",
            "delay_min_seconds": 0,
            "delay_max_seconds": 1,
            "clean_raw_on_start": True,
        },
        "transform": {
            "clean_output_on_start": True,
            "normalize_legal_headings": True,
            "libreoffice_cmd": "soffice",
        },
        "legal_chunking": {
            "max_tokens": 700,
            "overlap_tokens": 140,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/app.log",
            "max_bytes": 5242880,
            "backup_count": 5,
        },
        "ui": {
            "show_progress": True,
        },
        "rag": {
            "no_context_response": "Nao encontrei informacoes na base de conhecimento para responder a esta pergunta.",
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Aplica merge recursivo de dicionarios.

    Parameters
    ----------
    base : dict[str, Any]
        Configuracao base.
    override : dict[str, Any]
        Valores que devem sobrescrever a base.

    Returns
    -------
    dict[str, Any]
        Dicionario final mesclado.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(force_reload: bool = False) -> dict[str, Any]:
    """Carrega e cacheia configuracoes do arquivo `config.yaml`.

    Parameters
    ----------
    force_reload : bool, opcional
        Recarrega o arquivo mesmo com cache em memoria.

    Returns
    -------
    dict[str, Any]
        Configuracao consolidada.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    config = _default_config()
    config_path = _project_root() / "config.yaml"
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            config = _deep_merge(config, loaded)

    _CONFIG_CACHE = config
    return config


def get_config_value(path: str, default: Any = None) -> Any:
    """Obtem valor de configuracao via caminho pontuado.

    Parameters
    ----------
    path : str
        Caminho no formato `secao.chave.subchave`.
    default : Any, opcional
        Valor retornado caso o caminho nao exista.

    Returns
    -------
    Any
        Valor localizado na configuracao ou `default`.
    """
    current: Any = load_config()
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
