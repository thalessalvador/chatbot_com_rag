"""Utilitários de configuração centralizada do projeto.

Este módulo carrega o arquivo `config.yaml` da raiz do projeto e fornece
acesso tipado às configurações para todos os componentes (pipeline, app, RAG,
logging), evitando dispersão de parâmetros no `.env`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None


def _project_root() -> Path:
    """Retorna o diretório raiz do projeto.

    Parameters
    ----------
    None
        Função sem parâmetros.

    Returns
    -------
    Path
        Caminho absoluto da raiz do projeto.
    """
    return Path(__file__).resolve().parent.parent


def _default_config() -> dict[str, Any]:
    """Constrói configuração padrão quando `config.yaml` não existe.

    Parameters
    ----------
    None
        Função sem parâmetros.

    Returns
    -------
    dict[str, Any]
        Dicionário com valores padrão seguros.
    """
    return {
        "llm": {
            "provider": "ollama",
            "google_model": "gemini-2.5-flash",
            "google_temperature": 0.0,
            "ollama_model": "ministral-3:14b",
            "ollama_base_url": "http://localhost:11434",
            "ollama_temperature": 0.0,
            "ollama_num_ctx": 2048,
            "ollama_num_predict": 384,
            "ollama_num_gpu": 1,
            "ollama_num_thread": 0,
            "ollama_keep_alive": "30m",
        },
        "embeddings": {
            "model": "all-mpnet-base-v2",
            "device": "cpu",
            "batch_size": 64,
        },
        "retrieval": {
            "top_k": 5,
            "rrf_k": 60,
            "chroma_add_batch_size": 512,
        },
        "scraping": {
            "base_url": "https://appasp.economia.go.gov.br/pareceres/",
            "max_docs": 100,
            "timeout_seconds": 30,
            "user_agent": "chatbot-rag-scraper/1.0",
            "delay_min_seconds": 3,
            "delay_max_seconds": 10,
            "clean_raw_on_start": True,
        },
        "transform": {
            "clean_output_on_start": True,
            "normalize_legal_headings": True,
            "libreoffice_cmd": "soffice",
        },
        "legal_chunking": {
            "max_tokens": 700,
            "overlap_tokens": 80,
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
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Aplica merge recursivo de dicionários.

    Parameters
    ----------
    base : dict[str, Any]
        Configuração base.
    override : dict[str, Any]
        Valores que devem sobrescrever a base.

    Returns
    -------
    dict[str, Any]
        Dicionário final mesclado.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(force_reload: bool = False) -> dict[str, Any]:
    """Carrega e cacheia configurações do arquivo `config.yaml`.

    Parameters
    ----------
    force_reload : bool, opcional
        Recarrega o arquivo mesmo com cache em memória.

    Returns
    -------
    dict[str, Any]
        Configuração consolidada.
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
    """Obtém valor de configuração via caminho pontuado.

    Parameters
    ----------
    path : str
        Caminho no formato `secao.chave.subchave`.
    default : Any, opcional
        Valor retornado caso o caminho não exista.

    Returns
    -------
    Any
        Valor localizado na configuração ou `default`.
    """
    current: Any = load_config()
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

