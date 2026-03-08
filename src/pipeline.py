import argparse
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, urlsplit, urlunsplit

# Desativa telemetria do Chroma de forma explícita para evitar logs de PostHog.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
import requests
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from src.app_config import get_config_value
    from src.logging_config import get_logger
except ImportError:
    from app_config import get_config_value
    from logging_config import get_logger

logger = get_logger(__name__)

# Configurações de caminhos
RAW_DIR = Path("data/raw")
TRANSFORMATION_DIR = Path("data/transformation")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")

SCRAPING_MANIFEST_PATH = RAW_DIR / "scraping_manifest.jsonl"
TRANSFORM_MANIFEST_PATH = TRANSFORMATION_DIR / "transform_manifest.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
CHROMA_DB_DIR = str(INDEX_DIR / "chroma_db")
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

# Configurações de modelos
EMBEDDING_MODEL = get_config_value("embeddings.model", "all-mpnet-base-v2")
EMBEDDING_DEVICE = get_config_value("embeddings.device", "cpu")

LEGAL_SECTION_PATTERNS = [
    r"^\s*#+\s*(EMENTA|RELAT[ÓO]RIO|FUNDAMENTA[ÇC][ÃA]O|CONCLUS[ÃA]O|DISPOSITIVO)\b",
    r"^\s*[IVXLC]+\s*[-–]\s*(RELAT[ÓO]RIO|FUNDAMENTA[ÇC][ÃA]O|CONCLUS[ÃA]O|DISPOSITIVO)\b",
    r"^\s*(EMENTA|RELAT[ÓO]RIO|FUNDAMENTA[ÇC][ÃA]O|CONCLUS[ÃA]O|DISPOSITIVO)\s*[:\-]\s*",
]

TRIBUTOS_KEYWORDS = [
    "ICMS",
    "IPVA",
    "ITCD",
    "ISS",
    "IPI",
    "PIS",
    "COFINS",
    "FUNDEINFRA",
]


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
        "SCRAPING_MAX_DOCS": "scraping.max_docs",
        "SCRAPING_TIMEOUT_SECONDS": "scraping.timeout_seconds",
        "SCRAPING_DELAY_MIN_SECONDS": "scraping.delay_min_seconds",
        "SCRAPING_DELAY_MAX_SECONDS": "scraping.delay_max_seconds",
        "LEGAL_CHUNK_MAX_TOKENS": "legal_chunking.max_tokens",
        "LEGAL_CHUNK_OVERLAP_TOKENS": "legal_chunking.overlap_tokens",
        "EMBEDDING_BATCH_SIZE": "embeddings.batch_size",
        "CHROMA_ADD_BATCH_SIZE": "retrieval.chroma_add_batch_size",
    }
    try:
        return int(get_config_value(mapping.get(var_name, ""), default_value))
    except (TypeError, ValueError):
        return default_value


def _get_bool_env(var_name, default_value):
    """Lê uma variável booleana de ambiente com fallback seguro.

    Parameters
    ----------
    var_name : str
        Nome da variável de ambiente.
    default_value : bool
        Valor padrão quando a variável não existe.

    Returns
    -------
    bool
        Valor booleano convertido para uso interno.
    """
    mapping = {
        "SCRAPING_CLEAN_RAW_ON_START": "scraping.clean_raw_on_start",
        "TRANSFORM_CLEAN_OUTPUT_ON_START": "transform.clean_output_on_start",
        "TRANSFORM_NORMALIZE_LEGAL_HEADINGS": "transform.normalize_legal_headings",
        "SHOW_PROGRESS": "ui.show_progress",
    }
    raw = get_config_value(mapping.get(var_name, ""), default_value)
    if raw is None:
        return default_value
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_embedding_device(requested_device):
    """Resolve o dispositivo de embeddings com fallback seguro para CPU.

    Parameters
    ----------
    requested_device : str
        Dispositivo solicitado no ambiente (`cpu` ou `cuda`).

    Returns
    -------
    str
        Dispositivo efetivo para o SentenceTransformer.
    """
    device = (requested_device or "cpu").strip().lower()
    if device != "cuda":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    logger.warning("EMBEDDING_DEVICE=cuda, mas PyTorch sem CUDA. Usando CPU.")
    return "cpu"


def _ensure_dirs():
    """Garante que os diretórios de dados existam.

    Parameters
    ----------
    None
        Esta função não recebe parâmetros.

    Returns
    -------
    None
        Cria os diretórios necessários do pipeline.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMATION_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _clean_raw_dir():
    """Remove arquivos antigos da pasta `data/raw` antes do scraping.

    Parameters
    ----------
    None
        Esta função usa o diretório global `RAW_DIR`.

    Returns
    -------
    int
        Quantidade de arquivos removidos.
    """
    removed = 0
    for file_path in RAW_DIR.glob("*"):
        if file_path.is_file() and file_path.name != ".gitkeep":
            file_path.unlink(missing_ok=True)
            removed += 1
    return removed


def _clean_transformation_dir():
    """Remove artefatos antigos da pasta `data/transformation` antes da transformação.

    Parameters
    ----------
    None
        Esta função usa o diretório global `TRANSFORMATION_DIR`.

    Returns
    -------
    int
        Quantidade de arquivos removidos.
    """
    removed = 0
    for path in TRANSFORMATION_DIR.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            path.unlink(missing_ok=True)
            removed += 1

    # Remove diretórios temporários vazios (ex.: _tmp_converted).
    for path in sorted(TRANSFORMATION_DIR.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    return removed


def _safe_filename_from_url(url):
    """Gera nome de arquivo seguro com base na URL.

    Parameters
    ----------
    url : str
        URL de origem do arquivo.

    Returns
    -------
    str
        Nome seguro para uso no sistema de arquivos.
    """
    parsed = urlparse(url)
    name = Path(parsed.path).name or "documento"
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return clean


def _build_download_filename(doc_id, url):
    """Monta nome de arquivo no formato id_nomeoriginal.extensão.

    Parameters
    ----------
    doc_id : str
        Identificador único do documento.
    url : str
        URL original do arquivo a ser baixado.

    Returns
    -------
    str
        Nome de arquivo seguro para salvar em disco.
    """
    safe_name = _safe_filename_from_url(url)
    stem = Path(safe_name).stem or "arquivo"
    extension = Path(safe_name).suffix or ".bin"
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    stem = stem[:120]
    return f"{doc_id}_{stem}{extension}"


def _normalize_document_url(base_url, href):
    """Normaliza links de documentos para URL HTTP válida.

    Parameters
    ----------
    base_url : str
        URL base da página de scraping.
    href : str
        Valor bruto de `href` encontrado no HTML.

    Returns
    -------
    str
        URL absoluta normalizada e com path codificado.
    """
    raw = (href or "").strip()
    if not raw:
        return ""

    # Corrige separadores de diretório e remove barras duplicadas no início.
    raw = raw.replace("\\", "/")
    raw = re.sub(r"/{2,}", "/", raw)

    absolute = urljoin(base_url, raw)
    split = urlsplit(absolute)

    # Remove duplicidade comum do tipo /pareceres/pareceres/arquivos/...
    normalized_path = re.sub(
        r"^/pareceres/pareceres/",
        "/pareceres/",
        split.path,
        flags=re.IGNORECASE,
    )

    # Codifica apenas o caminho, preservando barras.
    quoted_path = quote(normalized_path, safe="/")
    return urlunsplit((split.scheme, split.netloc, quoted_path, split.query, split.fragment))


def _doc_id_from_url(url):
    """Gera doc_id determinístico com base na URL.

    Parameters
    ----------
    url : str
        URL do documento.

    Returns
    -------
    str
        Identificador único determinístico.
    """
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _write_jsonl(path, rows):
    """Escreve uma lista de registros JSONL no caminho informado.

    Parameters
    ----------
    path : Path
        Caminho de saída do arquivo JSONL.
    rows : list[dict]
        Registros que serão serializados.

    Returns
    -------
    None
        Persiste os registros em disco.
    """
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    """Lê um arquivo JSONL para lista de dicionários.

    Parameters
    ----------
    path : Path
        Caminho do arquivo JSONL.

    Returns
    -------
    list[dict]
        Lista de registros carregados do arquivo.
    """
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _discover_document_links(base_url, timeout_seconds, user_agent):
    """Descobre links de documentos na página base de pareceres.

    Parameters
    ----------
    base_url : str
        URL base da página de pareceres.
    timeout_seconds : int
        Tempo máximo de espera para requisições HTTP.
    user_agent : str
        User-Agent utilizado nas requisições.

    Returns
    -------
    list[dict]
        Lista de documentos candidatos com URL e metadados iniciais.
    """
    headers = {
        "User-Agent": user_agent,
        "Referer": base_url,
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }
    logger.info("Iniciando descoberta de links em %s", base_url)

    response = requests.get(base_url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    docs = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        full_url = _normalize_document_url(base_url, href)
        if not full_url:
            continue
        label = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()

        if full_url in seen:
            continue

        looks_like_document = bool(
            re.search(r"\.(pdf|doc|docx|txt|html?)($|\?)", full_url, re.IGNORECASE)
        )
        looks_like_parecer = "parecer" in full_url.lower() or "parecer" in label.lower()

        if looks_like_document or looks_like_parecer:
            seen.add(full_url)
            docs.append(
                {
                    "url": full_url,
                    "titulo": label or Path(urlparse(full_url).path).name or "Parecer sem título",
                    "fonte": full_url,
                    "data": None,
                    "tipo": "parecer",
                }
            )

    logger.info("Descoberta concluída: %s links candidatos.", len(docs))
    return docs


def run_scraping():
    """Executa scraping da página oficial e baixa documentos originais.

    Parameters
    ----------
    None
        Esta função usa variáveis de ambiente para configuração.

    Returns
    -------
    None
        Baixa arquivos para `data/raw` e registra manifesto de scraping.
    """
    _ensure_dirs()

    base_url = get_config_value("scraping.base_url", "https://appasp.economia.go.gov.br/pareceres/")
    max_docs = _get_int_env("SCRAPING_MAX_DOCS", 100)
    timeout_seconds = _get_int_env("SCRAPING_TIMEOUT_SECONDS", 30)
    user_agent = get_config_value("scraping.user_agent", "chatbot-rag-scraper/1.0")

    logger.info("Iniciando etapa de scraping.")
    logger.info("Configuração scraping: base_url=%s max_docs=%s", base_url, max_docs)

    if _get_bool_env("SCRAPING_CLEAN_RAW_ON_START", False):
        removed = _clean_raw_dir()
        logger.info("Limpeza de raw ativada: %s arquivo(s) removido(s) em %s.", removed, RAW_DIR)

    try:
        discovered = _discover_document_links(base_url, timeout_seconds, user_agent)
    except Exception as exc:
        logger.exception("Falha ao descobrir links de documentos: %s", exc)
        return

    discovered = discovered[:max_docs]
    delay_min_seconds = _get_int_env("SCRAPING_DELAY_MIN_SECONDS", 3)
    delay_max_seconds = _get_int_env("SCRAPING_DELAY_MAX_SECONDS", 10)
    if delay_max_seconds < delay_min_seconds:
        delay_max_seconds = delay_min_seconds

    headers = {
        "User-Agent": user_agent,
        "Referer": base_url,
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }
    manifest_rows = []

    for idx, entry in enumerate(
        tqdm(discovered, desc="Baixando documentos", disable=not _get_bool_env("SHOW_PROGRESS", True))
    ):
        url = entry["url"]
        doc_id = _doc_id_from_url(url)
        local_name = _build_download_filename(doc_id, url)
        output_path = RAW_DIR / local_name

        row = {
            "doc_id": doc_id,
            "titulo": entry["titulo"],
            "fonte": entry["fonte"],
            "data": entry["data"],
            "tipo": entry["tipo"],
            "source_url": url,
            "local_path": str(output_path.as_posix()),
            "status": "pending",
            "erro": None,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if idx > 0:
                delay = random.uniform(delay_min_seconds, delay_max_seconds)
                logger.info(
                    "Aguardando %.2fs antes do próximo download (controle anti-bloqueio).",
                    delay,
                )
                time.sleep(delay)

            logger.info("Baixando documento: %s", url)
            resp = requests.get(url, headers=headers, timeout=timeout_seconds)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            row["status"] = "downloaded"
            logger.info("Download concluído: %s -> %s", url, output_path)
        except Exception as exc:
            row["status"] = "error"
            row["erro"] = str(exc)
            logger.error("Erro ao baixar %s: %s", url, exc)

        manifest_rows.append(row)

    _write_jsonl(SCRAPING_MANIFEST_PATH, manifest_rows)
    ok_count = sum(1 for r in manifest_rows if r["status"] == "downloaded")
    err_count = sum(1 for r in manifest_rows if r["status"] == "error")
    logger.info(
        "Scraping finalizado. downloaded=%s error=%s manifesto=%s",
        ok_count,
        err_count,
        SCRAPING_MANIFEST_PATH,
    )


def _extract_markdown_from_result(result):
    """Extrai texto markdown de retorno do MarkItDown.

    Parameters
    ----------
    result : object
        Objeto retornado pelo conversor MarkItDown.

    Returns
    -------
    str
        Conteúdo markdown extraído do objeto.
    """
    for attr in ("text_content", "markdown", "text", "content"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return str(result)


def _convert_docx_with_python_docx(input_path):
    """Converte arquivo DOCX para markdown simples via `python-docx`.

    Parameters
    ----------
    input_path : Path
        Caminho do arquivo `.docx`.

    Returns
    -------
    str
        Texto convertido em markdown básico.
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "Dependência 'python-docx' não encontrada. Instale com: pip install python-docx"
        ) from exc

    doc = Document(str(input_path))
    parts = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        # Heurística simples para headings em caixa alta.
        if len(text) <= 120 and text.upper() == text and re.search(r"[A-ZÁÉÍÓÚÇ]", text):
            parts.append(f"## {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _resolve_soffice_command():
    """Resolve o executável do LibreOffice (`soffice`) disponível no sistema.

    Parameters
    ----------
    None
        Usa `transform.libreoffice_cmd` do `config.yaml` e o PATH atual do sistema.

    Returns
    -------
    str
        Caminho do executável `soffice` encontrado.
    """
    configured = str(get_config_value("transform.libreoffice_cmd", "soffice")).strip() or "soffice"
    candidates = [configured]
    if os.name == "nt":
        if not configured.lower().endswith(".exe"):
            candidates.append(f"{configured}.exe")
        if not configured.lower().endswith(".com"):
            candidates.append(f"{configured}.com")

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(
        "Executável do LibreOffice não encontrado. "
        "Configure transform.libreoffice_cmd no config.yaml ou adicione soffice ao PATH."
    )


def _convert_legacy_office_file(input_path):
    """Converte formatos legados do Office para formato moderno via LibreOffice.

    Parameters
    ----------
    input_path : Path
        Caminho do arquivo legado (`.doc`, `.xls`, `.ppt`).

    Returns
    -------
    Path
        Caminho do arquivo convertido gerado na pasta temporária de transformação.
    """
    suffix = input_path.suffix.lower()
    target_extension = {
        ".doc": "docx",
        ".xls": "xlsx",
        ".ppt": "pptx",
    }.get(suffix)

    if not target_extension:
        raise ValueError(f"Extensão legada não suportada para conversão automática: {suffix}")

    soffice_cmd = _resolve_soffice_command()
    conversion_dir = TRANSFORMATION_DIR / "_tmp_converted"
    conversion_dir.mkdir(parents=True, exist_ok=True)

    command = [
        soffice_cmd,
        "--headless",
        "--convert-to",
        target_extension,
        "--outdir",
        str(conversion_dir),
        str(input_path),
    ]
    logger.info("Convertendo arquivo legado via LibreOffice: %s", input_path)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        stdout_text = (result.stdout or "").strip()
        raise RuntimeError(
            "Falha na conversão via LibreOffice. "
            f"stdout={stdout_text} stderr={stderr_text}"
        )

    converted_path = conversion_dir / f"{input_path.stem}.{target_extension}"
    if not converted_path.exists():
        raise FileNotFoundError(
            f"Arquivo convertido não encontrado após LibreOffice: {converted_path}"
        )

    logger.info("Conversão legada concluída: %s -> %s", input_path, converted_path)
    return converted_path


def _build_markdown_front_matter(meta):
    """Monta front matter YAML com metadados mínimos do documento.

    Parameters
    ----------
    meta : dict
        Metadados do documento para serialização.

    Returns
    -------
    str
        String em formato front matter YAML.
    """
    lines = [
        "---",
        f"doc_id: {meta.get('doc_id', '')}",
        f"titulo: {json.dumps(meta.get('titulo', ''), ensure_ascii=False)}",
        f"assunto: {json.dumps(meta.get('assunto', ''), ensure_ascii=False)}",
        f"fonte: {json.dumps(meta.get('fonte', ''), ensure_ascii=False)}",
        f"data: {json.dumps(meta.get('data', ''), ensure_ascii=False)}",
        f"tipo: {json.dumps(meta.get('tipo', 'parecer'), ensure_ascii=False)}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _infer_document_title(markdown_body, fallback_title):
    """Infere o título jurídico principal a partir do conteúdo markdown.

    Parameters
    ----------
    markdown_body : str
        Corpo do documento convertido para markdown.
    fallback_title : str
        Título de fallback (normalmente derivado do arquivo/origem).

    Returns
    -------
    str
        Título inferido do documento, ou `fallback_title` quando não encontrado.
    """
    if not markdown_body or not markdown_body.strip():
        return fallback_title

    lines = [line.strip() for line in markdown_body.splitlines() if line.strip()]

    # Prioriza headings explícitos que contenham a identificação do parecer.
    for line in lines[:120]:
        cleaned = re.sub(r"^#+\s*", "", line).strip()
        if re.search(r"\bPARECER\b", cleaned, flags=re.IGNORECASE):
            return cleaned

    # Fallback: primeiro heading markdown disponível.
    for line in lines[:120]:
        if line.startswith("#"):
            cleaned = re.sub(r"^#+\s*", "", line).strip()
            if cleaned:
                return cleaned

    return fallback_title


def _canonicalize_legal_heading(raw_heading):
    """Normaliza um heading jurídico para um formato canônico.

    Parameters
    ----------
    raw_heading : str
        Texto bruto do possível heading jurídico.

    Returns
    -------
    str | None
        Heading canônico quando reconhecido; caso contrário, `None`.
    """
    text = re.sub(r"^#+\s*", "", (raw_heading or "").strip())
    # Remove marcadores de ênfase comuns no markdown.
    text = re.sub(r"[*_`]+", "", text).strip()
    text = text.strip("\"'“”")
    text = re.sub(r"\s+", " ", text).strip()
    upper = text.upper()

    dash = r"[-–—â€“]"
    if re.fullmatch(rf"(?:[IVXLC]+\s*{dash}\s*)?(?:DA\s+)?RELAT[ÓO]RIO[:\-]?", upper):
        return "I - RELATÓRIO"
    if re.fullmatch(rf"(?:[IVXLC]+\s*{dash}\s*)?(?:DA\s+)?FUNDAMENTA[ÇC][ÃA]O[:\-]?", upper):
        return "II - FUNDAMENTAÇÃO"
    if re.fullmatch(rf"(?:[IVXLC]+\s*{dash}\s*)?(?:DA\s+)?CONCLUS[ÃA]O[:\-]?", upper):
        return "III - CONCLUSÃO"
    if re.fullmatch(rf"(?:[IVXLC]+\s*{dash}\s*)?EMENTA[:\-]?", upper):
        return "EMENTA"
    if re.fullmatch(rf"(?:[IVXLC]+\s*{dash}\s*)?DISPOSITIVO[:\-]?", upper):
        return "DISPOSITIVO"
    return None


def _normalize_legal_headings(markdown_body):
    """Padroniza seções jurídicas como headings Markdown no texto convertido.

    Parameters
    ----------
    markdown_body : str
        Texto markdown do documento antes da normalização.

    Returns
    -------
    str
        Texto markdown com headings jurídicos padronizados.
    """
    if not markdown_body:
        return markdown_body

    normalized_lines = []
    kept_main_title = False
    canonical_set = {
        "I - RELATÓRIO",
        "II - FUNDAMENTAÇÃO",
        "III - CONCLUSÃO",
        "EMENTA",
        "DISPOSITIVO",
    }
    for line in markdown_body.splitlines():
        stripped = line.strip()
        canonical = _canonicalize_legal_heading(stripped)
        if canonical:
            normalized_lines.append(f"## {canonical}")
            continue

        # Rebaixa headings nível 2 que não são estruturais (evita conflito com seção pai).
        if stripped.startswith("## "):
            heading_text = re.sub(r"^##\s*", "", stripped).strip()
            heading_text = re.sub(r"[*_`]+", "", heading_text).strip()
            heading_text = heading_text.strip("\"'“”")
            heading_text = re.sub(r"\s+", " ", heading_text).strip()
            canonical_heading = _canonicalize_legal_heading(heading_text)

            if canonical_heading in canonical_set:
                normalized_lines.append(f"## {canonical_heading}")
                continue

            if not kept_main_title and re.search(r"\bPARECER\b", heading_text, flags=re.IGNORECASE):
                normalized_lines.append(f"## {heading_text}")
                kept_main_title = True
                continue

            normalized_lines.append(f"### {heading_text}")
        else:
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def run_transform():
    """Converte arquivos originais para Markdown usando MarkItDown.

    Parameters
    ----------
    None
        A função depende do manifesto de scraping em `data/raw`.

    Returns
    -------
    None
        Gera `.md` em `data/transformation` e manifesto de transformação.
    """
    _ensure_dirs()
    logger.info("Iniciando etapa de transformação para Markdown.")
    if _get_bool_env("TRANSFORM_CLEAN_OUTPUT_ON_START", True):
        removed = _clean_transformation_dir()
        logger.info(
            "Limpeza da pasta de transformação concluída. arquivos_removidos=%s",
            removed,
        )

    records = _read_jsonl(SCRAPING_MANIFEST_PATH)
    if not records:
        logger.error("Manifesto de scraping ausente ou vazio: %s", SCRAPING_MANIFEST_PATH)
        return

    converter = None
    try:
        from markitdown import MarkItDown
        converter = MarkItDown()
    except ImportError:
        logger.warning(
            "markitdown não disponível; será usado fallback para DOCX via python-docx quando possível."
        )
    output_rows = []
    stats = {
        "markitdown_ok": 0,
        "fallback_python_docx": 0,
        "legacy_converted_via_libreoffice": 0,
        "errors": 0,
    }

    for rec in tqdm(records, desc="Transformando para Markdown", disable=not _get_bool_env("SHOW_PROGRESS", True)):
        if rec.get("status") != "downloaded":
            continue

        input_path = Path(rec["local_path"])
        # Mantém o padrão id_nomeoriginal também na transformação.
        output_stem = input_path.stem
        md_path = TRANSFORMATION_DIR / f"{output_stem}.md"

        out = {
            "doc_id": rec["doc_id"],
            "titulo": rec.get("titulo"),
            "fonte": rec.get("fonte"),
            "data": rec.get("data"),
            "tipo": rec.get("tipo", "parecer"),
            "raw_path": str(input_path.as_posix()),
            "md_path": str(md_path.as_posix()),
            "status": "pending",
            "erro": None,
            "metodo_transformacao": None,
            "transformed_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            logger.info("Transformando arquivo para MD: %s", input_path)
            source_for_conversion = input_path
            suffix = source_for_conversion.suffix.lower()

            if suffix in {".doc", ".xls", ".ppt"}:
                source_for_conversion = _convert_legacy_office_file(input_path)
                suffix = source_for_conversion.suffix.lower()
                stats["legacy_converted_via_libreoffice"] += 1

            if converter is not None:
                try:
                    result = converter.convert(str(source_for_conversion))
                    markdown_body = _extract_markdown_from_result(result)
                    out["metodo_transformacao"] = "markitdown"
                    stats["markitdown_ok"] += 1
                except Exception as exc:
                    # Fallback dedicado para DOCX quando plugin/docx do MarkItDown não está presente.
                    if suffix == ".docx":
                        logger.info(
                            "MarkItDown indisponível/incompatível para %s; usando fallback python-docx.",
                            source_for_conversion,
                        )
                        markdown_body = _convert_docx_with_python_docx(source_for_conversion)
                        out["metodo_transformacao"] = "python-docx-fallback"
                        stats["fallback_python_docx"] += 1
                    else:
                        raise
            else:
                if suffix == ".docx":
                    markdown_body = _convert_docx_with_python_docx(source_for_conversion)
                    out["metodo_transformacao"] = "python-docx-fallback"
                    stats["fallback_python_docx"] += 1
                else:
                    raise ValueError(
                        "Sem MarkItDown e sem fallback para este formato. "
                        "Instale dependências de conversão ou use arquivos .docx."
                    )

            if _get_bool_env("TRANSFORM_NORMALIZE_LEGAL_HEADINGS", True):
                markdown_body = _normalize_legal_headings(markdown_body)

            out["titulo"] = _infer_document_title(markdown_body, out.get("titulo"))
            out["assunto"] = _extract_assunto(markdown_body)
            markdown_content = _build_markdown_front_matter(out) + markdown_body
            md_path.write_text(markdown_content, encoding="utf-8")
            out["status"] = "ok"
            logger.info("Transformação concluída: %s", md_path)
        except Exception as exc:
            out["status"] = "error"
            out["erro"] = str(exc)
            stats["errors"] += 1
            logger.error("Erro na transformação de %s: %s", input_path, exc)

        output_rows.append(out)

    _write_jsonl(TRANSFORM_MANIFEST_PATH, output_rows)
    ok_count = sum(1 for r in output_rows if r["status"] == "ok")
    err_count = sum(1 for r in output_rows if r["status"] == "error")
    logger.info(
        "Transformação finalizada. ok=%s error=%s manifesto=%s",
        ok_count,
        err_count,
        TRANSFORM_MANIFEST_PATH,
    )
    logger.info(
        "Resumo de métodos de transformação: markitdown_ok=%s fallback_python_docx=%s legacy_convertidos=%s errors=%s",
        stats["markitdown_ok"],
        stats["fallback_python_docx"],
        stats["legacy_converted_via_libreoffice"],
        stats["errors"],
    )


def _parse_front_matter(markdown_text):
    """Extrai front matter YAML simples de conteúdo Markdown.

    Parameters
    ----------
    markdown_text : str
        Conteúdo completo do arquivo markdown.

    Returns
    -------
    tuple[dict, str]
        Par (metadados extraídos, corpo sem front matter).
    """
    lines = markdown_text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, markdown_text

    metadata = {}
    end_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_index = i
            break
        if ":" in lines[i]:
            key, value = lines[i].split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')

    if end_index == -1:
        return {}, markdown_text

    body = "\n".join(lines[end_index + 1 :]).strip()
    return metadata, body


def _normalize_section_name(raw_title):
    """Normaliza nome de seção jurídica para rótulo canônico.

    Parameters
    ----------
    raw_title : str
        Título original da seção.

    Returns
    -------
    str
        Nome normalizado da seção.
    """
    text = raw_title.upper()
    if "EMENTA" in text:
        return "ementa"
    if "RELAT" in text:
        return "relatorio"
    if "FUNDAMENT" in text:
        return "fundamentacao"
    if "CONCLUS" in text:
        return "conclusao"
    if "DISPOSITIVO" in text:
        return "dispositivo"
    return "secao_geral"


def _detect_legal_sections(markdown_body):
    """Segmenta markdown em seções jurídicas por headings e padrões.

    Parameters
    ----------
    markdown_body : str
        Corpo do markdown sem front matter.

    Returns
    -------
    list[dict]
        Lista de seções com título, rótulo e texto da seção.
    """
    lines = markdown_body.splitlines()
    sections = []
    current_title = "Texto Geral"
    current_label = "secao_geral"
    current_buffer = []

    def flush_current():
        text = "\n".join(current_buffer).strip()
        if text:
            sections.append(
                {
                    "titulo_secao": current_title,
                    "secao": current_label,
                    "texto": text,
                }
            )

    for line in lines:
        stripped = line.strip()

        matched_heading = False
        for pattern in LEGAL_SECTION_PATTERNS:
            if re.search(pattern, stripped, flags=re.IGNORECASE):
                flush_current()
                current_title = stripped
                current_label = _normalize_section_name(stripped)
                current_buffer = []
                matched_heading = True
                break

        if matched_heading:
            continue

        if stripped.startswith("#"):
            flush_current()
            current_title = stripped.lstrip("#").strip()
            current_label = _normalize_section_name(current_title)
            current_buffer = []
            continue

        current_buffer.append(line)

    flush_current()
    if not sections:
        sections.append(
            {
                "titulo_secao": "Texto Geral",
                "secao": "secao_geral",
                "texto": markdown_body.strip(),
            }
        )
    return sections


def _safe_tokenize(text):
    """Tokeniza texto com fallback sem dependência de recursos externos.

    Parameters
    ----------
    text : str
        Texto de entrada para tokenização.

    Returns
    -------
    list[str]
        Lista de tokens normalizados.
    """
    try:
        import nltk

        return nltk.word_tokenize(text)
    except Exception:
        return re.findall(r"\w+", text, flags=re.UNICODE)


def _split_section_into_subchunks(section_text, max_tokens, overlap_tokens):
    """Subdivide seção textual longa em subchunks por contagem de tokens.

    Parameters
    ----------
    section_text : str
        Texto integral da seção jurídica.
    max_tokens : int
        Quantidade máxima de tokens por subchunk.
    overlap_tokens : int
        Sobreposição de tokens entre subchunks consecutivos.

    Returns
    -------
    list[str]
        Lista de textos de subchunks.
    """
    tokens = _safe_tokenize(section_text)
    if len(tokens) <= max_tokens:
        return [section_text.strip()]

    chunks = []
    start = 0
    step = max(1, max_tokens - max(0, overlap_tokens))
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        sub_tokens = tokens[start:end]
        chunks.append(" ".join(sub_tokens))
        if end >= len(tokens):
            break
        start += step
    return chunks


def _extract_normas_citadas(text):
    """Extrai referências normativas comuns de texto jurídico.

    Parameters
    ----------
    text : str
        Texto onde as normas serão buscadas.

    Returns
    -------
    list[str]
        Lista deduplicada de padrões normativos detectados.
    """
    patterns = [
        r"\bart\.?\s*\d+[A-Za-zº°\-]*",
        r"\blei\s*(n[º°.]*)?\s*\d+[./]?\d*",
        r"\bCTN\b",
        r"\bCF/88\b",
        r"\bConstitui[çc][ãa]o\b",
    ]
    matches = []
    for p in patterns:
        matches.extend(re.findall(p, text, flags=re.IGNORECASE))
    cleaned = []
    seen = set()
    for m in matches:
        val = re.sub(r"\s+", " ", m).strip()
        key = val.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(val)
    return cleaned


def _extract_tributos_citados(text):
    """Extrai tributos citados com base em lista de palavras-chave.

    Parameters
    ----------
    text : str
        Texto de entrada para identificação de tributos.

    Returns
    -------
    list[str]
        Lista de tributos encontrados no texto.
    """
    upper = text.upper()
    return [t for t in TRIBUTOS_KEYWORDS if t in upper]


def _build_enriched_text(metadata, section_text):
    """Monta texto enriquecido com cabeçalho semântico para embeddings.

    Parameters
    ----------
    metadata : dict
        Metadados do chunk e do documento.
    section_text : str
        Texto principal do chunk.

    Returns
    -------
    str
        Texto enriquecido para indexação vetorial.
    """
    normas = "; ".join(metadata.get("normas_citadas", [])) or "N/A"
    tributos = "; ".join(metadata.get("tributos_citados", [])) or "N/A"
    assunto = metadata.get("assunto") or "N/A"
    return (
        f"[Tipo: {metadata.get('tipo', 'parecer')}]\n"
        f"[Doc ID: {metadata.get('doc_id', 'N/A')}]\n"
        f"[Título: {metadata.get('titulo', 'N/A')}]\n"
        f"[Assunto: {assunto}]\n"
        f"[Seção: {metadata.get('secao', 'secao_geral')}]\n"
        f"[Normas citadas: {normas}]\n"
        f"[Tributos citados: {tributos}]\n\n"
        f"Texto:\n{section_text.strip()}"
    )


def _extract_assunto(markdown_body):
    """Extrai o campo `Assunto` do corpo Markdown quando presente.

    Parameters
    ----------
    markdown_body : str
        Texto do documento sem front matter.

    Returns
    -------
    str | None
        Texto do assunto identificado, ou `None` quando não encontrado.
    """
    for raw_line in markdown_body.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Remove bullets e ênfases simples antes da análise.
        cleaned = re.sub(r"^[-*]\s*", "", line)
        cleaned = re.sub(r"[*_`]+", "", cleaned).strip()

        match = re.match(r"(?i)^assunto\s*:\s*(.+)$", cleaned)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def run_ingest():
    """Executa ingestão a partir de Markdown com chunking jurídico semântico.

    Parameters
    ----------
    None
        Esta função utiliza os artefatos gerados em `data/transformation`.

    Returns
    -------
    None
        Gera `data/processed/chunks.json` com chunks enriquecidos e metadados.
    """
    _ensure_dirs()
    logger.info("Iniciando ingestão e chunking jurídico a partir de Markdown.")

    records = _read_jsonl(TRANSFORM_MANIFEST_PATH)
    if not records:
        logger.error("Manifesto de transformação ausente ou vazio: %s", TRANSFORM_MANIFEST_PATH)
        return

    max_tokens = _get_int_env("LEGAL_CHUNK_MAX_TOKENS", 700)
    overlap_tokens = _get_int_env("LEGAL_CHUNK_OVERLAP_TOKENS", 80)

    all_chunks = []
    show_progress = _get_bool_env("SHOW_PROGRESS", True)

    for rec in tqdm(records, desc="Gerando chunks", disable=not show_progress):
        if rec.get("status") != "ok":
            continue

        md_path = Path(rec["md_path"])
        if not md_path.exists():
            logger.warning("Arquivo markdown não encontrado, ignorando: %s", md_path)
            continue

        markdown_text = md_path.read_text(encoding="utf-8")
        front_matter, body = _parse_front_matter(markdown_text)
        doc_meta = {
            "doc_id": front_matter.get("doc_id") or rec.get("doc_id"),
            "titulo": front_matter.get("titulo") or rec.get("titulo"),
            "fonte": front_matter.get("fonte") or rec.get("fonte"),
            "data": front_matter.get("data") or rec.get("data"),
            "tipo": front_matter.get("tipo") or rec.get("tipo", "parecer"),
            "assunto": front_matter.get("assunto"),
        }
        if not doc_meta.get("assunto"):
            doc_meta["assunto"] = _extract_assunto(body)

        sections = _detect_legal_sections(body)
        chunk_counter = 0
        for section in sections:
            subchunks = _split_section_into_subchunks(
                section["texto"], max_tokens=max_tokens, overlap_tokens=overlap_tokens
            )
            for sub in subchunks:
                normas = _extract_normas_citadas(sub)
                tributos = _extract_tributos_citados(sub)
                chunk_id = f"{doc_meta['doc_id']}#chunk_{chunk_counter:04d}"
                chunk_counter += 1

                metadata = {
                    **doc_meta,
                    "chunk_id": chunk_id,
                    "secao": section["secao"],
                    "titulo_secao": section["titulo_secao"],
                    "topico": section["titulo_secao"],
                    "normas_citadas": normas,
                    "tributos_citados": tributos,
                }

                enriched_text = _build_enriched_text(metadata, sub)
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_meta["doc_id"],
                        "texto": enriched_text,
                        "texto_bruto": sub,
                        "metadados": metadata,
                    }
                )

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info("Chunks gerados em %s (total=%s).", CHUNKS_PATH, len(all_chunks))


def run_index():
    """Executa indexação híbrida Dense + Sparse dos chunks processados.

    Parameters
    ----------
    None
        A função usa os chunks de `data/processed/chunks.json`.

    Returns
    -------
    None
        Atualiza os artefatos de índice vetorial e BM25 em `data/index`.
    """
    logger.info("Iniciando indexação híbrida.")

    if not CHUNKS_PATH.exists():
        logger.error("Chunks não encontrados. Rode ingestão primeiro: %s", CHUNKS_PATH)
        return

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Gerando embeddings dense e salvando no ChromaDB.")
    effective_embedding_device = _resolve_embedding_device(EMBEDDING_DEVICE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=effective_embedding_device)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        chroma_client.delete_collection(name="pareceres_tributarios")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="pareceres_tributarios")

    def _sanitize_metadata_for_chroma(metadata):
        """Converte metadados para tipos aceitos pelo ChromaDB.

        Parameters
        ----------
        metadata : dict
            Dicionário bruto de metadados do chunk.

        Returns
        -------
        dict
            Dicionário com valores apenas em `str`, `int`, `float` ou `bool`.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            elif isinstance(value, (list, tuple, set)):
                sanitized[key] = "; ".join(str(v) for v in value)
            else:
                sanitized[key] = str(value)
        return sanitized

    textos = [c["texto"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadados = [_sanitize_metadata_for_chroma(c["metadados"]) for c in chunks]

    embedding_batch_size = _get_int_env("EMBEDDING_BATCH_SIZE", 64)
    show_progress = _get_bool_env("SHOW_PROGRESS", True)
    embeddings = model.encode(
        textos,
        batch_size=embedding_batch_size,
        show_progress_bar=show_progress,
    ).tolist()

    chroma_add_batch_size = _get_int_env("CHROMA_ADD_BATCH_SIZE", 512)
    total_items = len(ids)
    for start in tqdm(
        range(0, total_items, chroma_add_batch_size),
        desc="Salvando no ChromaDB",
        disable=not show_progress,
    ):
        end = min(start + chroma_add_batch_size, total_items)
        collection.add(
            embeddings=embeddings[start:end],
            documents=textos[start:end],
            metadatas=metadados[start:end],
            ids=ids[start:end],
        )

    logger.info("Gerando índice sparse (BM25).")
    tokenized_corpus = [_safe_tokenize(doc.lower()) for doc in textos]
    bm25 = BM25Okapi(tokenized_corpus)

    with BM25_INDEX_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    logger.info("Indexação concluída com sucesso.")


def run_evaluate():
    """Executa avaliação do retriever (placeholder da fase atual).

    Parameters
    ----------
    None
        Esta etapa será expandida nas próximas fases do roadmap.

    Returns
    -------
    None
        Apenas registra status da implementação de avaliação.
    """
    logger.info("Executando avaliação do retriever no Golden Set...")
    logger.info("Implementação de Recall@k pendente de conexão com Golden Set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG - Projeto Final")
    parser.add_argument(
        "--step",
        choices=["scraping", "transform", "ingest", "index", "evaluate"],
        required=True,
    )
    args = parser.parse_args()

    if args.step == "scraping":
        run_scraping()
    elif args.step == "transform":
        run_transform()
    elif args.step == "ingest":
        run_ingest()
    elif args.step == "index":
        run_index()
    elif args.step == "evaluate":
        run_evaluate()
