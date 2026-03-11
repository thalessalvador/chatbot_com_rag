import json
import pickle
import os

# Desativa telemetria do Chroma de forma explÃ­cita para evitar logs de PostHog.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
import re
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
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

# FIX: Garante que os pacotes de tokenizaÃ§Ã£o do NLTK sÃ£o descarregados no contentor do chatbot
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()
logger = get_logger(__name__)
NO_CONTEXT_RESPONSE = str(
    get_config_value(
        "rag.no_context_response",
        "Não encontrei informações na base de conhecimento para responder a esta pergunta.",
    )
)


def _extract_response_text(response):
    """Extrai texto da resposta do LangChain com fallback seguro."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content)


def _is_no_context_response(answer_text):
    """Valida se a resposta retornada Ã© a resposta padrÃ£o de ausÃªncia de contexto."""
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
                "titulo": metadata.get("titulo", "Documento sem tÃ­tulo"),
                "fonte": metadata.get("fonte", ""),
                "conteudo": chunk.get("texto", "") or "",
            }
        )
    return entries


def _build_context_with_trecho_aliases(alias_entries):
    """Monta contexto com delimitadores para facilitar parsing pelo LLM."""
    parts = []
    for entry in alias_entries:
        parts.append(
            "\n".join(
                [
                    "<trecho_disponivel>",
                    f"ID: [[{entry['alias']}]]",
                    f"chunk_id_real: {entry['chunk_id']}",
                    f"titulo: {entry['titulo']}",
                    f"fonte: {entry['fonte'] or 'N/A'}",
                    f"conteudo: {entry['conteudo']}",
                    "</trecho_disponivel>",
                ]
            )
        )
    return "\n\n".join(parts)


def _build_fontes_disponiveis(alias_entries):
    """Monta bloco resumido de fontes disponÃ­veis para prompt de correÃ§Ã£o."""
    lines = []
    for entry in alias_entries:
        lines.append(
            f"- [[{entry['alias']}]] | titulo={entry['titulo']} | fonte={entry['fonte'] or 'N/A'} | chunk_id={entry['chunk_id']}"
        )
    return "\n".join(lines)


def _build_trechos_disponiveis(alias_entries, preview_chars=350):
    """Monta bloco de trechos com resumo para tarefa de mapeamento de citaÃ§Ãµes."""
    lines = []
    for entry in alias_entries:
        preview = " ".join(entry["conteudo"].split())
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + "..."
        lines.append(f"- [[{entry['alias']}]]: {preview}")
    return "\n".join(lines)


def _has_trecho_alias_citation(answer_text):
    """Indica se o texto contÃ©m pelo menos uma citaÃ§Ã£o no formato [[TRECHO_n]]."""
    return bool(re.search(r"\[\[TRECHO_\d+\]\]", answer_text or ""))


def _convert_trecho_aliases_to_html(answer_text, alias_entries):
    """Converte [[TRECHO_n]] para citaÃ§Ã£o com link HTML e chunk_id."""
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
    """Anexa lista de fontes quando o modelo nÃ£o cita aliases vÃ¡lidos."""
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
    """LÃª uma variÃ¡vel de ambiente inteira com fallback seguro.

    Parameters
    ----------
    var_name : str
        Nome da variÃ¡vel de ambiente.
    default_value : int
        Valor padrÃ£o quando a variÃ¡vel nÃ£o existe ou Ã© invÃ¡lida.

    Returns
    -------
    int
        Valor inteiro vÃ¡lido para uso na configuraÃ§Ã£o.
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
    """LÃª uma variÃ¡vel de ambiente numÃ©rica (float) com fallback seguro.

    Parameters
    ----------
    var_name : str
        Nome da variÃ¡vel de ambiente.
    default_value : float
        Valor padrÃ£o quando a variÃ¡vel nÃ£o existe ou Ã© invÃ¡lida.

    Returns
    -------
    float
        Valor numÃ©rico vÃ¡lido para uso na configuraÃ§Ã£o.
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


def _safe_tokenize(text):
    """Tokeniza texto com fallback sem dependÃªncia de recursos externos.

    Parameters
    ----------
    text : str
        Texto de entrada para tokenizaÃ§Ã£o.

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
    """Normaliza notaÃ§Ã£o matemÃ¡tica estilo LaTeX para texto simples.

    Parameters
    ----------
    answer_text : str
        Texto de resposta gerado pelo LLM.

    Returns
    -------
    str
        Texto com expressÃµes matemÃ¡ticas em formato textual simples.
    """
    if not answer_text:
        return answer_text

    def _latex_to_plain(raw_text):
        """Converte expressÃ£o LaTeX simples para representaÃ§Ã£o textual."""
        parsed = raw_text

        # SÃ³ usa parser LaTeX quando hÃ¡ comando LaTeX explÃ­cito (ex.: \frac, \text).
        has_latex_command = bool(re.search(r"\\[A-Za-z]+", parsed))
        if LatexNodes2Text is not None and has_latex_command:
            try:
                # Protege porcentagens para nÃ£o serem tratadas como comentÃ¡rio LaTeX.
                parsed_for_latex = parsed.replace("%", r"\%")
                parsed = LatexNodes2Text().latex_to_text(parsed_for_latex)
            except Exception:
                # MantÃ©m fallback regex abaixo caso o parser falhe.
                parsed = raw_text

        # Remove wrappers comuns de bloco matemÃ¡tico.
        parsed = re.sub(r"^\s*\[\s*(.*?)\s*\]\s*$", r"\1", parsed, flags=re.DOTALL)
        parsed = parsed.strip()
        if parsed.startswith("[") and not parsed.endswith("]"):
            parsed = parsed.lstrip("[").strip()
        if parsed.endswith("]") and not parsed.startswith("["):
            parsed = parsed.rstrip("]").strip()

        # Remove apenas delimitadores de bloco matemÃ¡tico que contenham comando LaTeX.
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

        # Resolve \text{...} antes de tentar fraÃ§Ãµes genÃ©ricas.
        parsed = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", parsed)

        # FraÃ§Ãµes genÃ©ricas simples.
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
            "Ã—": " x ",
        }
        for src, dst in replacements.items():
            parsed = parsed.replace(src, dst)

        parsed = parsed.replace("{", "").replace("}", "")
        parsed = re.sub(r"[ \t]+", " ", parsed)
        parsed = re.sub(r"\n{3,}", "\n\n", parsed)
        return parsed.strip()

    text = answer_text
    latex_candidates = []

    # Captura blocos com comandos LaTeX para comparaÃ§Ã£o em modo debug.
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
    """Converte tabelas Markdown em lista de bullets legÃ­vel.

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
        """Indica se um trecho contÃ©m apenas traÃ§os/separadores visuais."""
        return bool(re.fullmatch(r"[\s\-\â€”\â€“\|\:]+", text or ""))

    while i < len(lines):
        line = lines[i]
        is_table_line = line.count("|") >= 2
        if not is_table_line:
            # Fallback para "linhas-tabela" sem cabeÃ§alho formal.
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

        # Primeira linha Ãºtil como cabeÃ§alho.
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
        Texto com Ãªnfases/tabelas malformadas reduzidas.
    """
    if not answer_text:
        return answer_text

    text = answer_text
    # Remove marcaÃ§Ãµes de Ãªnfase que frequentemente vÃªm desbalanceadas.
    text = text.replace("**", "")
    text = text.replace("__", "")

    # Normaliza separadores exagerados.
    text = re.sub(r"[â€”\-]{4,}", "â€”", text)

    # Corrige espaÃ§os comuns em moeda.
    text = re.sub(r"R\s*\$\s*", "R$ ", text)
    text = re.sub(r"R\$\s+", "R$ ", text)

    return text


def _log_llm_request(query, retrieved_chunks, context_str):
    """Registra no log o payload de recuperaÃ§Ã£o enviado ao LLM.

    Parameters
    ----------
    query : str
        Pergunta original do usuÃ¡rio.
    retrieved_chunks : list[dict]
        Chunks recuperados e selecionados para compor o contexto.
    context_str : str
        Contexto consolidado que serÃ¡ interpolado no prompt.

    Returns
    -------
    None
        Apenas emite logs para inspeÃ§Ã£o operacional.
    """
    logger.info("--- LOG DE RECUPERAÃ‡ÃƒO (Top-K) ---")
    logger.info("Pergunta enviada ao LLM: %s", query)
    logger.info("Quantidade de chunks recuperados: %s", len(retrieved_chunks))

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadados", {})
        chunk_id = chunk.get("chunk_id", "N/A")
        titulo = metadata.get("titulo", "Documento sem tÃ­tulo")
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
    """Registra o prompt completo enviado ao LLM.

    Parameters
    ----------
    prompt : str
        Prompt final jÃ¡ interpolado com contexto e pergunta.

    Returns
    -------
    None
        Apenas emite logs para inspeÃ§Ã£o detalhada.
    """
    logger.info("--- INICIO PROMPT COMPLETO ENVIADO AO LLM ---")
    logger.info("%s", prompt)
    logger.info("--- FIM PROMPT COMPLETO ENVIADO AO LLM ---")


class HybridRAG:
    """Orquestra o fluxo RAG hÃ­brido (Dense + Sparse) e a geraÃ§Ã£o de respostas.

    Esta classe centraliza:
    - carregamento dos Ã­ndices vetorial (ChromaDB) e BM25;
    - recuperaÃ§Ã£o hÃ­brida com fusÃ£o RRF;
    - geraÃ§Ã£o de resposta com grounding usando LLM configurÃ¡vel.
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
            Caminho para o arquivo pickle que contÃ©m o Ã­ndice BM25 e chunks.
        llm_provider : str | None, opcional
            Provedor de geraÃ§Ã£o (`google` ou `ollama`). Se None, lÃª de
            `LLM_PROVIDER` no ambiente.
        llm_model : str | None, opcional
            Nome do modelo do provedor selecionado. Se None, usa variÃ¡vel de
            ambiente correspondente.
        ollama_base_url : str | None, opcional
            URL base da API do Ollama. Se None, lÃª de `OLLAMA_BASE_URL`.

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
        embedding_model_name = get_config_value("embeddings.model", "all-mpnet-base-v2")
        embedding_device = get_config_value("embeddings.device", "cpu")
        effective_embedding_device = _resolve_embedding_device(embedding_device)
        self.rrf_k = _get_int_env("RRF_K", 60)
        self.default_top_k = _get_int_env("RETRIEVAL_TOP_K", 5)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=effective_embedding_device)
        
        # Sparse
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks_data = data['chunks']

        provider = (llm_provider or get_config_value("llm.provider", "google")).strip().lower()
        self.llm_provider = provider
        self.llm_model = None

        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "Para usar LLM_PROVIDER=google, instale: pip install langchain-google-genai"
                ) from exc
            model_name = llm_model or get_config_value("llm.google_model", "gemini-2.5-flash")
            temperature = _get_float_env("GOOGLE_TEMPERATURE", 0.0)
            self.llm_model = model_name
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        elif provider == "ollama":
            model_name = llm_model or get_config_value("llm.ollama_model", "llama3.1:8b")
            base_url = ollama_base_url or get_config_value("llm.ollama_base_url", "http://localhost:11434")
            ollama_temperature = _get_float_env("OLLAMA_TEMPERATURE", 0.0)
            ollama_num_ctx = _get_int_env("OLLAMA_NUM_CTX", 2048)
            ollama_num_predict = _get_int_env("OLLAMA_NUM_PREDICT", 384)
            ollama_num_gpu = _get_int_env("OLLAMA_NUM_GPU", 1)
            ollama_num_thread = _get_int_env("OLLAMA_NUM_THREAD", 0)
            ollama_keep_alive = get_config_value("llm.ollama_keep_alive", "30m")
            self.llm_model = model_name

            self.llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=ollama_temperature,
                num_ctx=ollama_num_ctx,
                num_predict=ollama_num_predict,
                num_gpu=ollama_num_gpu,
                num_thread=ollama_num_thread,
                keep_alive=ollama_keep_alive
            )
        else:
            raise ValueError(
                "LLM_PROVIDER invÃ¡lido. Use 'google' ou 'ollama'."
            )

    def retrieve(self, query, top_k=None):
        """Recupera os chunks mais relevantes via busca hÃ­brida com RRF.

        Parameters
        ----------
        query : str
            Pergunta do usuÃ¡rio utilizada para recuperar contexto.
        top_k : int, opcional
            Quantidade final de chunks retornados apÃ³s a fusÃ£o dos rankings.

        Returns
        -------
        list[dict]
            Lista de chunks recuperados. Cada item contÃ©m `chunk_id`, `doc_id`,
            `texto` e `metadados`.
        """
        
        if top_k is None:
            top_k = self.default_top_k

        # 1. RecuperaÃ§Ã£o Dense
        query_embedding = self.embedding_model.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 2 # Busca mais para fazer a fusÃ£o
        )
        dense_ids = dense_results['ids'][0]
        
        # 2. RecuperaÃ§Ã£o Sparse (BM25)
        tokenized_query = _safe_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Pega os top K * 2 Ã­ndices
        top_sparse_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        sparse_ids = [self.chunks_data[i]["chunk_id"] for i in top_sparse_idx]

        # 3. FusÃ£o RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        k_rrf = self.rrf_k
        
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
            
        for rank, doc_id in enumerate(sparse_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)

        # Ordena e pega os K finais
        final_top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        # Monta a lista de contextos recuperados
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks_data}
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
        fontes_disponiveis = _build_fontes_disponiveis(alias_entries)
        trechos_disponiveis = _build_trechos_disponiveis(alias_entries)

        _log_llm_request(query, retrieved_chunks, context_str)
        logger.info("LLM em uso | provider=%s | model=%s", self.llm_provider, self.llm_model)

        prompt_template = PromptTemplate(
            input_variables=["contexto", "pergunta", "resposta_sem_contexto"],
            template="""Você é um assistente tributário do Estado de Goiás.
Sua tarefa é responder à pergunta do utilizador APENAS com base no contexto fornecido.

REGRAS OBRIGATÓRIAS:
1. GROUNDING: Se a resposta não estiver no contexto, responda EXATAMENTE: "{resposta_sem_contexto}"
2. NÃO INVENTE: Não use conhecimentos externos à base fornecida. O foco não é aconselhamento jurídico genérico.
3. IDIOMA: Responda sempre em Português do Brasil (pt-BR).
4. FORMATAÇÃO: Não use LaTeX, MathJax, Markdown matemático nem expressões como \text{{...}}.
5. CONCISÃO: Seja objetivo. Responda em no máximo 8 bullets curtos.
6. TABELAS: Não use tabelas em Markdown (com `|`).

FORMATO OBRIGATÓRIO DE SAÍDA:
- Resposta direta: 1 a 2 bullets.
- Base legal: 2 a 4 bullets.
- Como aplicar: 2 a 3 bullets.
- CITAÇÕES: Sempre que afirmar algo, cite no final da frase usando APENAS o alias do trecho.
  Formato exato obrigatório: [[TRECHO_n]]
  Exemplo: "O ICMS é isento neste caso [[TRECHO_2]]."

EXEMPLO DE RESPOSTA CORRETA:
- Resposta direta: O contribuinte goiano tem isenção de ICMS na saída de frutas frescas [[TRECHO_1]].
- Base legal: Conforme o Artigo 7º do RCTE-GO [[TRECHO_2]].
- Como aplicar: Deve-se emitir a nota fiscal com CFOP adequado [[TRECHO_1]].

CONTEXTO RECUPERADO:
{contexto}

PERGUNTA: {pergunta}

LEMBRETE OBRIGATÓRIO ANTES DE RESPONDER:
- Você deve citar o alias [[TRECHO_n]] ao final de cada afirmação relevante.
- Não use negrito.
- Responda em bullets.
- Se a resposta não estiver no contexto, responda EXATAMENTE: "{resposta_sem_contexto}". Não use conhecimentos externos à base fornecida. O foco não é aconselhamento jurídico genérico.

RESPOSTA:""",
        )

        prompt = prompt_template.format(
            contexto=context_str,
            pergunta=query,
            resposta_sem_contexto=NO_CONTEXT_RESPONSE,
        )
        _log_full_prompt(prompt)
        response = self.llm.invoke(prompt)
        raw_answer = _extract_response_text(response)

        logger.info("--- INICIO RESPOSTA BRUTA DO LLM ---")
        logger.info("%s", raw_answer)
        logger.info("--- FIM RESPOSTA BRUTA DO LLM ---")

        if _is_no_context_response(raw_answer):
            return NO_CONTEXT_RESPONSE

        answer_with_aliases = raw_answer
        if not _has_trecho_alias_citation(raw_answer):
            logger.info("Resposta sem aliases [[TRECHO_n]] na primeira passada. Iniciando prompt de correção.")
            fix_prompt_template = PromptTemplate(
                input_variables=["pergunta", "fontes", "trechos", "resposta_original"],
                template="""Você é um assistente tributário do Estado de Goiás.
Sua única tarefa é revisar a RESPOSTA ORIGINAL e inserir os identificadores [[TRECHO_n]] onde houver suporte nos documentos fornecidos.
Se uma frase não tiver evidência nos trechos, remova-a completamente.
Não altere o texto além de adicionar as etiquetas [[TRECHO_n]] ou remover frases órfãs.

Regras obrigatórias:
1. Não invente informação nova e não use conhecimento externo.
2. Use somente as fontes listadas em FONTES DISPONÍVEIS.
3. Use apenas os TRECHOS DISPONÍVEIS para validar as afirmações.
4. Inclua citação no formato [[TRECHO_n]] ao final de cada bullet relevante.
5. Não use HTML nesta etapa.
6. Retorne apenas a resposta final.

Formato obrigatório desta saída:
- Resposta direta: 1 a 2 bullets.
- Base legal: 2 a 4 bullets.
- Como aplicar: 2 a 3 bullets.

PERGUNTA:
{pergunta}

FONTES DISPONÍVEIS:
{fontes}

TRECHOS DISPONÍVEIS:
{trechos}

RESPOSTA ORIGINAL:
{resposta_original}

RESPOSTA FINAL:""",
            )
            fix_prompt = fix_prompt_template.format(
                pergunta=query,
                fontes=fontes_disponiveis,
                trechos=trechos_disponiveis,
                resposta_original=raw_answer,
            )
            logger.info("--- INICIO PROMPT DE CORRECAO DE CITACOES ---")
            logger.info("%s", fix_prompt)
            logger.info("--- FIM PROMPT DE CORRECAO DE CITACOES ---")
            fix_response = self.llm.invoke(fix_prompt)
            answer_with_aliases = _extract_response_text(fix_response)
            logger.info("--- INICIO RESPOSTA BRUTA DA CORRECAO ---")
            logger.info("%s", answer_with_aliases)
            logger.info("--- FIM RESPOSTA BRUTA DA CORRECAO ---")

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
