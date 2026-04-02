import json
import logging
import pickle
import os

# Desativa telemetria do Chroma de forma explícita para evitar logs de PostHog.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
import re
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer
import nltk
from dotenv import load_dotenv
try:
    from pylatexenc.latex2text import LatexNodes2Text
except ImportError:
    LatexNodes2Text = None
try:
    from src.app_config import get_config_value
    from src.logging_config import get_logger
except ImportError:
    from app_config import get_config_value
    from logging_config import get_logger

# FIX: Garante que os pacotes de tokenização do NLTK são descarregados no contentor do chatbot
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()
logger = get_logger(__name__)
NO_CONTEXT_RESPONSE = str(
    get_config_value("rag.no_context_response")
)


def _extract_response_text(response):
    """Extrai texto da resposta do LangChain com fallback seguro."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content)


def _is_no_context_response(answer_text):
    """Valida se a resposta retornada é a resposta padrão de ausência de contexto."""
    if not answer_text:
        return False
    normalized = answer_text.strip().strip('"').strip()
    return normalized == NO_CONTEXT_RESPONSE


def _build_trecho_alias_entries(retrieved_chunks):
    """Cria aliases curtos (TRECHO_n) para os chunks recuperados."""
    entries = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadados", {})
        entries.append(
            {
                "alias": f"TRECHO_{idx}",
                "chunk_id": chunk.get("chunk_id", ""),
                "titulo": metadata.get("titulo", "Documento sem título"),
                "fonte": metadata.get("fonte", ""),
                "conteudo": chunk.get("texto", "") or "",
            }
        )
    return entries


def _build_context_with_trecho_aliases(alias_entries):
    """Monta contexto compacto para o LLM (menos tokens que tags XML longas)."""
    parts = []
    for entry in alias_entries:
        cab = (
            f"[[{entry['alias']}]] "
            f"titulo={entry['titulo']} | fonte={entry['fonte'] or 'N/A'}"
        )
        parts.append(f"{cab}\n{entry['conteudo']}")
    return "\n\n".join(parts)


def _build_trechos_disponiveis(alias_entries, preview_chars=280):
    """Monta bloco de trechos com resumo para tarefa de mapeamento de citações."""
    lines = []
    for entry in alias_entries:
        preview = " ".join(entry["conteudo"].split())
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + "..."
        lines.append(f"- [[{entry['alias']}]]: {preview}")
    return "\n".join(lines)


def _has_trecho_alias_citation(answer_text):
    """Indica se o texto contém pelo menos uma citação no formato [[TRECHO_n]]."""
    return bool(re.search(r"\[\[TRECHO_\d+\]\]", answer_text or ""))


def _convert_trecho_aliases_to_html(answer_text, alias_entries):
    """Converte [[TRECHO_n]] para citação com link HTML e chunk_id."""
    if not answer_text:
        return answer_text

    alias_map = {entry["alias"]: entry for entry in alias_entries}

    def _replace(match):
        alias = match.group(1)
        entry = alias_map.get(alias)
        if not entry:
            return match.group(0)
        titulo = entry["titulo"]
        fonte = entry["fonte"]
        chunk_id = entry["chunk_id"]
        if fonte:
            return (
                f'[<a href="{fonte}" target="_blank" rel="noopener noreferrer">{titulo}</a> {chunk_id}]'
            )
        return f"[{titulo} {chunk_id}]"

    return re.sub(r"\[\[(TRECHO_\d+)\]\]", _replace, answer_text)


def _append_fallback_sources(answer_text, alias_entries):
    """Anexa lista de fontes quando o modelo não cita aliases válidos."""
    if not answer_text or "Fontes consultadas:" in answer_text:
        return answer_text

    lines = [answer_text.rstrip(), "", "Fontes consultadas:", ""]
    for entry in alias_entries:
        titulo = entry["titulo"]
        chunk_id = entry["chunk_id"]
        fonte = entry["fonte"]
        if fonte:
            lines.append(
                f'- [<a href="{fonte}" target="_blank" rel="noopener noreferrer">{titulo}</a> {chunk_id}]'
            )
        else:
            lines.append(f"- [{titulo} {chunk_id}]")
    return "\n".join(lines)

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
        "RRF_K": "retrieval.rrf_k",
        "RETRIEVAL_TOP_K": "retrieval.top_k",
        "OLLAMA_NUM_CTX": "llm.ollama_num_ctx",
        "OLLAMA_NUM_PREDICT": "llm.ollama_num_predict",
        "OLLAMA_NUM_GPU": "llm.ollama_num_gpu",
        "OLLAMA_NUM_THREAD": "llm.ollama_num_thread",
    }
    try:
        return int(get_config_value(mapping.get(var_name, ""), default_value))
    except (TypeError, ValueError):
        return default_value

def _get_float_env(var_name, default_value):
    """Lê uma variável de ambiente numérica (float) com fallback seguro.

    Parameters
    ----------
    var_name : str
        Nome da variável de ambiente.
    default_value : float
        Valor padrão quando a variável não existe ou é inválida.

    Returns
    -------
    float
        Valor numérico válido para uso na configuração.
    """
    mapping = {
        "GOOGLE_TEMPERATURE": "llm.google_temperature",
        "OLLAMA_TEMPERATURE": "llm.ollama_temperature",
    }
    try:
        return float(get_config_value(mapping.get(var_name, ""), default_value))
    except (TypeError, ValueError):
        return default_value

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


def _get_bool_config(path, default_value):
    """Lê um valor booleano da configuração com fallback seguro.

    Parameters
    ----------
    path : str
        Caminho pontuado no `config.yaml`.
    default_value : bool
        Valor padrão usado quando a chave não existe.

    Returns
    -------
    bool
        Valor booleano normalizado para uso interno.
    """
    raw = get_config_value(path, default_value)
    if raw is None:
        return default_value
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_int_config(path, default_value):
    """Lê um valor inteiro da configuração com fallback seguro.

    Parameters
    ----------
    path : str
        Caminho pontuado no `config.yaml`.
    default_value : int
        Valor padrão usado quando a chave não existe ou é inválida.

    Returns
    -------
    int
        Valor inteiro normalizado para uso interno.
    """
    try:
        return int(get_config_value(path, default_value))
    except (TypeError, ValueError):
        return default_value


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
        return nltk.word_tokenize(text)
    except Exception:
        return re.findall(r"\w+", text, flags=re.UNICODE)


def _normalize_math_notation(answer_text):
    """Normaliza notação matemática estilo LaTeX para texto simples.

    Parameters
    ----------
    answer_text : str
        Texto de resposta gerado pelo LLM.

    Returns
    -------
    str
        Texto com expressões matemáticas em formato textual simples.
    """
    if not answer_text:
        return answer_text

    def _latex_to_plain(raw_text):
        """Converte expressão LaTeX simples para representação textual."""
        parsed = raw_text

        # Só usa parser LaTeX quando há comando LaTeX explícito (ex.: \frac, \text).
        has_latex_command = bool(re.search(r"\\[A-Za-z]+", parsed))
        if LatexNodes2Text is not None and has_latex_command:
            try:
                # Protege porcentagens para não serem tratadas como comentário LaTeX.
                parsed_for_latex = parsed.replace("%", r"\%")
                parsed = LatexNodes2Text().latex_to_text(parsed_for_latex)
            except Exception:
                # Mantém fallback regex abaixo caso o parser falhe.
                parsed = raw_text

        # Remove wrappers comuns de bloco matemático.
        parsed = re.sub(r"^\s*\[\s*(.*?)\s*\]\s*$", r"\1", parsed, flags=re.DOTALL)
        parsed = parsed.strip()
        if parsed.startswith("[") and not parsed.endswith("]"):
            parsed = parsed.lstrip("[").strip()
        if parsed.endswith("]") and not parsed.startswith("["):
            parsed = parsed.rstrip("]").strip()

        # Remove apenas delimitadores de bloco matemático que contenham comando LaTeX.
        parsed = re.sub(
            r"\[\s*([^\]]*\\[A-Za-z]+[^\]]*)\s*\]",
            lambda m: m.group(1),
            parsed,
            flags=re.DOTALL,
        )

        # Casos comuns com \frac envolvendo \text{...}.
        parsed = re.sub(
            r"\\frac\s*\{\s*\\text\s*\{([^{}]+)\}\s*\}\s*\{\s*\\text\s*\{([^{}]+)\}\s*\}",
            r"(\1 / \2)",
            parsed,
        )

        # Resolve \text{...} antes de tentar frações genéricas.
        parsed = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", parsed)

        # Frações genéricas simples.
        frac_pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
        while frac_pattern.search(parsed):
            parsed = frac_pattern.sub(r"(\1 / \2)", parsed)

        # Fallbacks para sintaxes incompletas.
        parsed = re.sub(r"\\frac\s*\(([^()]+)\)\s*\(([^()]+)\)", r"(\1 / \2)", parsed)
        parsed = re.sub(r"\\frac\b", " dividido por ", parsed)

        replacements = {
            r"\times": " x ",
            r"\cdot": " * ",
            r"\left": "",
            r"\right": "",
            r"\(": "(",
            r"\)": ")",
            r"\[": "[",
            r"\]": "]",
            "×": " x ",
        }
        for src, dst in replacements.items():
            parsed = parsed.replace(src, dst)

        parsed = parsed.replace("{", "").replace("}", "")
        parsed = re.sub(r"[ \t]+", " ", parsed)
        parsed = re.sub(r"\n{3,}", "\n\n", parsed)
        return parsed.strip()

    text = answer_text
    latex_candidates = []

    # Captura blocos com comandos LaTeX para comparação em modo debug.
    for m in re.finditer(r"\[[^\]]*\\[A-Za-z]+[^\]]*\]", text, flags=re.DOTALL):
        snippet = m.group(0).strip()
        if snippet:
            latex_candidates.append(snippet)
    for m in re.finditer(r"\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}", text):
        snippet = m.group(0).strip()
        if snippet:
            latex_candidates.append(snippet)
    for m in re.finditer(r"\$[^$]*\\[A-Za-z][^$]*\$", text):
        snippet = m.group(0).strip()
        if snippet:
            latex_candidates.append(snippet)

    text = _latex_to_plain(text)

    return text


def _normalize_markdown_tables(answer_text):
    """Converte tabelas Markdown em lista de bullets legível.

    Parameters
    ----------
    answer_text : str
        Texto de resposta gerado pelo LLM.

    Returns
    -------
    str
        Texto sem tabela Markdown, com itens em bullets.
    """
    if not answer_text:
        return answer_text

    lines = answer_text.splitlines()
    output = []
    i = 0

    def _is_dash_only(text):
        """Indica se um trecho contém apenas traços/separadores visuais."""
        return bool(re.fullmatch(r"[\s\-\â€”\â€“\|\:]+", text or ""))

    while i < len(lines):
        line = lines[i]
        is_table_line = line.count("|") >= 2
        if not is_table_line:
            # Fallback para "linhas-tabela" sem cabeçalho formal.
            if line.count("|") >= 1:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                parts = [p for p in parts if not _is_dash_only(p)]
                if len(parts) >= 2:
                    output.append("- " + " | ".join(parts))
                    i += 1
                    continue
            output.append(line)
            i += 1
            continue

        # Coleta bloco de tabela consecutivo.
        block = []
        while i < len(lines) and lines[i].count("|") >= 2:
            block.append(lines[i].strip())
            i += 1

        # Remove linhas separadoras tipo |---|---|
        filtered = []
        for row in block:
            row_no_pipes = re.sub(r"[\|\-\â€”\â€“]", "", row).strip()
            if row_no_pipes:
                filtered.append(row)

        if len(filtered) < 2:
            output.extend(block)
            continue

        # Primeira linha útil como cabeçalho.
        header_cells = [c.strip() for c in filtered[0].strip("|").split("|")]
        data_rows = filtered[1:]

        for row in data_rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            cells = [c for c in cells if not _is_dash_only(c)]
            parts = []
            for idx, cell in enumerate(cells):
                if not cell:
                    continue
                key = header_cells[idx] if idx < len(header_cells) else f"Campo {idx+1}"
                parts.append(f"{key}: {cell}")
            if parts:
                output.append("- " + " | ".join(parts))

    return "\n".join(output)


def _normalize_markdown_artifacts(answer_text):
    """Remove artefatos de Markdown que prejudicam legibilidade no Streamlit.

    Parameters
    ----------
    answer_text : str
        Texto de resposta gerado pelo LLM.

    Returns
    -------
    str
        Texto com ênfases/tabelas malformadas reduzidas.
    """
    if not answer_text:
        return answer_text

    text = answer_text
    # Remove marcações de ênfase que frequentemente vêm desbalanceadas.
    text = text.replace("**", "")
    text = text.replace("__", "")

    # Normaliza separadores exagerados.
    text = re.sub(r"[â€”\-]{4,}", "â€”", text)

    # Corrige espaços comuns em moeda.
    text = re.sub(r"R\s*\$\s*", "R$ ", text)
    text = re.sub(r"R\$\s+", "R$ ", text)

    return text


def _log_llm_request(query, retrieved_chunks, context_str):
    """Registra no log o payload de recuperação enviado ao LLM.

    Parameters
    ----------
    query : str
        Pergunta original do usuário.
    retrieved_chunks : list[dict]
        Chunks recuperados e selecionados para compor o contexto.
    context_str : str
        Contexto consolidado que será interpolado no prompt.

    Returns
    -------
    None
        Apenas emite logs para inspeção operacional.
    """
    logger.info("--- LOG DE RECUPERAÇÃO (Top-K) ---")
    logger.info("Pergunta enviada ao LLM: %s", query)
    logger.info("Quantidade de chunks recuperados: %s", len(retrieved_chunks))

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadados", {})
        chunk_id = chunk.get("chunk_id", "N/A")
        titulo = metadata.get("titulo", "Documento sem título")
        secao = metadata.get("secao", "secao_geral")

        source_text = chunk.get("texto_bruto") or chunk.get("texto", "")
        preview = " ".join(str(source_text).split())
        if len(preview) > 280:
            preview = preview[:280] + "..."

        logger.info("Resultado %s | ID: %s", idx, chunk_id)
        logger.info("Titulo: %s | Secao: %s", titulo, secao)
        logger.info("Texto enviado ao LLM: %s", preview)

    logger.info("Contexto consolidado (tamanho em caracteres): %s", len(context_str))


def _log_full_prompt(prompt):
    """Registra tamanho do prompt; texto completo só em nível DEBUG (evita I/O lento)."""
    logger.info("Prompt LLM (principal) | chars=%s", len(prompt))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("--- PROMPT COMPLETO ---\n%s\n--- FIM ---", prompt)


class HybridRAG:
    """Orquestra o fluxo RAG híbrido (Dense + Sparse) e a geração de respostas.

    Esta classe centraliza:
    - carregamento dos índices vetorial (ChromaDB) e BM25;
    - recuperação híbrida com fusão RRF;
    - geração de resposta com grounding usando LLM configurável.
    """

    def __init__(
        self,
        chroma_path="data/index/chroma_db",
        bm25_path="data/index/bm25_index.pkl",
        llm_provider=None,
        llm_model=None,
        ollama_base_url=None
    ):
        """Inicializa os componentes do sistema RAG e o provedor de LLM.

        Parameters
        ----------
        chroma_path : str, opcional
            Caminho para o banco persistente do ChromaDB com embeddings.
        bm25_path : str, opcional
            Caminho para o arquivo pickle que contém o índice BM25 e chunks.
        llm_provider : str | None, opcional
            Provedor de geração (`google` ou `ollama`). Se None, lê de
            `LLM_PROVIDER` no ambiente.
        llm_model : str | None, opcional
            Nome do modelo do provedor selecionado. Se None, usa variável de
            ambiente correspondente.
        ollama_base_url : str | None, opcional
            URL base da API do Ollama. Se None, lê de `OLLAMA_BASE_URL`.

        Returns
        -------
        None
            Construtor da classe; inicializa atributos internos.
        """
        # Dense
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(name="pareceres_tributarios")
        embedding_model_name = get_config_value("embeddings.model")
        embedding_device = get_config_value("embeddings.device")
        effective_embedding_device = _resolve_embedding_device(embedding_device)
        self.rrf_k = _get_int_env("RRF_K", 60)
        self.default_top_k = _get_int_env("RETRIEVAL_TOP_K", 5)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=effective_embedding_device)

        reranker_enabled = _get_bool_config("retrieval.reranking.enabled", False)
        self.reranker = None
        self.reranker_enabled = reranker_enabled
        self.reranker_model_name = None
        self.reranker_candidate_pool_size = max(
            self.default_top_k,
            _get_int_config("retrieval.reranking.candidate_pool_size", 20),
        )
        if reranker_enabled:
            reranker_model_name = get_config_value("retrieval.reranking.model")
            reranker_device = _resolve_embedding_device(
                get_config_value("retrieval.reranking.device", "cpu")
            )
            self.reranker = CrossEncoder(reranker_model_name, device=reranker_device)
            self.reranker_model_name = reranker_model_name
            logger.info(
                "Reranker habilitado | model=%s | device=%s | candidate_pool_size=%s",
                reranker_model_name,
                reranker_device,
                self.reranker_candidate_pool_size,
            )
        
        # Sparse
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks_data = data['chunks']

        provider = (llm_provider or get_config_value("llm.provider")).strip().lower()
        self.llm_provider = provider
        self.llm_model = None

        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "Para usar LLM_PROVIDER=google, instale: pip install langchain-google-genai"
                ) from exc
            model_name = llm_model or get_config_value("llm.google_model")
            temperature = _get_float_env("GOOGLE_TEMPERATURE", 0.0)
            self.llm_model = model_name
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        elif provider == "ollama":
            model_name = llm_model or get_config_value("llm.ollama_model")
            base_url = ollama_base_url or get_config_value("llm.ollama_base_url")
            ollama_temperature = _get_float_env("OLLAMA_TEMPERATURE", 0.0)
            ollama_num_ctx = _get_int_env("OLLAMA_NUM_CTX", 2048)
            ollama_num_predict = _get_int_env("OLLAMA_NUM_PREDICT", 384)
            ollama_num_gpu = _get_int_env("OLLAMA_NUM_GPU", 1)
            ollama_num_thread = _get_int_env("OLLAMA_NUM_THREAD", 0)
            ollama_keep_alive = get_config_value("llm.ollama_keep_alive")
            self.llm_model = model_name

            timeout_sec = get_config_value("llm.ollama_client_timeout_seconds")
            client_kwargs = {}
            try:
                if timeout_sec is not None and float(timeout_sec) > 0:
                    client_kwargs["timeout"] = float(timeout_sec)
            except (TypeError, ValueError):
                pass

            self.llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=ollama_temperature,
                num_ctx=ollama_num_ctx,
                num_predict=ollama_num_predict,
                num_gpu=ollama_num_gpu,
                num_thread=ollama_num_thread,
                keep_alive=ollama_keep_alive,
                client_kwargs=client_kwargs,
            )
        else:
            raise ValueError(
                "LLM_PROVIDER inválido. Use 'google' ou 'ollama'."
            )

    def _rerank_hybrid_candidates(self, query, candidate_ids, chunk_by_id, top_k):
        """Reordena candidatos híbridos usando um modelo de reranking.

        Parameters
        ----------
        query : str
            Pergunta original do usuário.
        candidate_ids : list[str]
            Identificadores dos chunks candidatos vindos da fusão RRF.
        chunk_by_id : dict[str, dict]
            Mapeamento de `chunk_id` para o chunk completo.
        top_k : int
            Quantidade final de resultados a retornar.

        Returns
        -------
        list[str]
            Lista de `chunk_id` reranqueados, limitada a `top_k`.
        """
        if not self.reranker or not candidate_ids:
            return candidate_ids[:top_k]

        pairs = []
        valid_ids = []
        for chunk_id in candidate_ids:
            chunk = chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            chunk_text = chunk.get("texto_bruto") or chunk.get("texto") or ""
            if not chunk_text.strip():
                continue
            valid_ids.append(chunk_id)
            pairs.append([query, chunk_text])

        if not pairs:
            return candidate_ids[:top_k]

        scores = self.reranker.predict(pairs)
        scored_ids = sorted(
            zip(valid_ids, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        logger.info(
            "Reranking aplicado | candidatos=%s | top_k=%s | melhores=%s",
            len(valid_ids),
            top_k,
            [chunk_id for chunk_id, _ in scored_ids[:top_k]],
        )
        return [chunk_id for chunk_id, _ in scored_ids[:top_k]]

    def retrieve(self, query, top_k=None, mode="hybrid"):
        """Recupera os chunks mais relevantes.

        Parameters
        ----------
        query : str
            Pergunta do usuário utilizada para recuperar contexto.
        top_k : int, opcional
            Quantidade final de chunks retornados.
        mode : str, opcional
            Modo de busca a ser utilizado: "dense", "sparse" ou "hybrid". Padrão é "hybrid".

        Returns
        -------
        list[dict]
            Lista de chunks recuperados. Cada item contém `chunk_id`, `doc_id`,
            `texto` e `metadados`.
        """

        if top_k is None:
            top_k = self.default_top_k
        dense_sparse_pool_size = top_k * 2
        hybrid_candidate_pool_size = (
            max(top_k, self.reranker_candidate_pool_size)
            if self.reranker_enabled
            else dense_sparse_pool_size
        )

        # 1. Recuperação Dense
        query_embedding = self.embedding_model.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=max(dense_sparse_pool_size, hybrid_candidate_pool_size),
        )
        dense_ids = dense_results['ids'][0]

        # 2. Recuperação Sparse (BM25)
        tokenized_query = _safe_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:max(dense_sparse_pool_size, hybrid_candidate_pool_size)]
        sparse_ids = [self.chunks_data[i]["chunk_id"] for i in top_sparse_idx]

        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks_data}

        # 3. Seleção do Modo
        if mode == "dense":
            final_top_ids = dense_ids[:top_k]
        elif mode == "sparse":
            final_top_ids = sparse_ids[:top_k]
        else:
            # Fusão RRF original
            rrf_scores = {}
            k_rrf = self.rrf_k
            
            for rank, doc_id in enumerate(dense_ids):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
                
            for rank, doc_id in enumerate(sparse_ids):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)

            fused_candidate_ids = sorted(
                rrf_scores.keys(),
                key=lambda x: rrf_scores[x],
                reverse=True,
            )[:hybrid_candidate_pool_size]

            if self.reranker_enabled:
                final_top_ids = self._rerank_hybrid_candidates(
                    query=query,
                    candidate_ids=fused_candidate_ids,
                    chunk_by_id=chunk_by_id,
                    top_k=top_k,
                )
            else:
                final_top_ids = fused_candidate_ids[:top_k]

        # Monta a lista de contextos recuperados
        recovered_chunks = [chunk_by_id[cid] for cid in final_top_ids if cid in chunk_by_id]
        return recovered_chunks

    def generate_answer(self, query, retrieved_chunks):
        """Gera resposta ancorada no contexto recuperado com regras de citação.

        Parameters
        ----------
        query : str
            Pergunta do usuário.
        retrieved_chunks : list[dict]
            Trechos recuperados pelo método `retrieve`, usados como contexto.

        Returns
        -------
        str
            Resposta textual do LLM, restrita ao contexto recuperado.
        """

        if not retrieved_chunks:
            return NO_CONTEXT_RESPONSE

        alias_entries = _build_trecho_alias_entries(retrieved_chunks)
        context_str = _build_context_with_trecho_aliases(alias_entries)
        trechos_disponiveis = _build_trechos_disponiveis(alias_entries)

        _log_llm_request(query, retrieved_chunks, context_str)
        logger.info("LLM em uso | provider=%s | model=%s", self.llm_provider, self.llm_model)

        prompt_template = PromptTemplate(
            input_variables=["contexto", "pergunta", "resposta_sem_contexto"],
            template="""Assistente tributário (Goiás). Responda só com base nos TRECHOS. Idioma: pt-BR.

Regras:
- Sem informação nos trechos → responda exatamente: {resposta_sem_contexto}
- Não use conhecimento externo. Sem LaTeX/MathJax. Seja objetivo.
- Cada afirmação sustentada termina com [[TRECHO_n]] (n = índice do trecho).
- Responda sempre em Português do Brasil (pt-br)
Saída:
- Resposta direta: 1–2 parágrafos.
- Base legal: até 4 parágrafos com norma e [[TRECHO_n]].

TRECHOS:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:""",
        )

        prompt = prompt_template.format(
            contexto=context_str,
            pergunta=query,
            resposta_sem_contexto=NO_CONTEXT_RESPONSE,
        )
        _log_full_prompt(prompt)
        logger.info("LLM Ollama: passo 1/2 — geração principal (aguarde)")
        response = self.llm.invoke(prompt)
        raw_answer = _extract_response_text(response)

        logger.info(
            "--- RESPOSTA BRUTA (passo 1) | chars=%s ---",
            len(raw_answer or ""),
        )
        logger.info("%s", raw_answer)
        logger.info("--- FIM RESPOSTA BRUTA (passo 1) ---")

        if _is_no_context_response(raw_answer):
            return NO_CONTEXT_RESPONSE

        answer_with_aliases = raw_answer
        if not _has_trecho_alias_citation(raw_answer):
            logger.info("Resposta sem aliases [[TRECHO_n]] na primeira passada. Iniciando prompt de correção.")
            fix_prompt_template = PromptTemplate(
                input_variables=["pergunta", "trechos", "resposta_original"],
                template="""Revise a RESPOSTA ORIGINAL: acrescente [[TRECHO_n]] no fim das frases apoiadas nos trechos; remova frases sem evidência. Não invente conteúdo. pt-BR. Devolva só a resposta final.

TRECHOS (resumo):
{trechos}

PERGUNTA: {pergunta}

RESPOSTA ORIGINAL:
{resposta_original}

RESPOSTA FINAL:""",
            )
            fix_prompt = fix_prompt_template.format(
                pergunta=query,
                trechos=trechos_disponiveis,
                resposta_original=raw_answer,
            )
            logger.info(
                "Prompt correção citações | chars=%s",
                len(fix_prompt),
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "--- PROMPT CORREÇÃO ---\n%s\n--- FIM ---",
                    fix_prompt,
                )
            logger.info(
                "LLM Ollama: passo 2/2 — correção de citações [[TRECHO_n]] (pode demorar)"
            )
            fix_response = self.llm.invoke(fix_prompt)
            answer_with_aliases = _extract_response_text(fix_response)
            logger.info(
                "--- RESPOSTA BRUTA (passo 2) | chars=%s ---",
                len(answer_with_aliases or ""),
            )
            logger.info("%s", answer_with_aliases)
            logger.info("--- FIM RESPOSTA BRUTA (passo 2) ---")

            if _is_no_context_response(answer_with_aliases):
                return NO_CONTEXT_RESPONSE

        answer_with_links = _convert_trecho_aliases_to_html(answer_with_aliases, alias_entries)
        if '[<a href="' not in answer_with_links:
            logger.info("Sem citação válida após passadas. Aplicando fallback de fontes.")
            answer_with_links = _append_fallback_sources(answer_with_links, alias_entries)

        answer_with_links = _normalize_math_notation(answer_with_links)
        answer_with_links = _normalize_markdown_artifacts(answer_with_links)
        answer_with_links = _normalize_markdown_tables(answer_with_links)
        return answer_with_links