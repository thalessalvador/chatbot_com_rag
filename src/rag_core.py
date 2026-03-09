import json
import pickle
import os

# Desativa telemetria do Chroma de forma explícita para evitar logs de PostHog.
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

# FIX: Garante que os pacotes de tokenização do NLTK são descarregados no contentor do chatbot
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()
logger = get_logger(__name__)

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
        return bool(re.fullmatch(r"[\s\-\—\–\|\:]+", text or ""))

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
            row_no_pipes = re.sub(r"[\|\-\—\–]", "", row).strip()
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

        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "Para usar LLM_PROVIDER=google, instale: pip install langchain-google-genai"
                ) from exc
            model_name = llm_model or get_config_value("llm.google_model", "gemini-2.5-flash")
            temperature = _get_float_env("GOOGLE_TEMPERATURE", 0.0)
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
                "LLM_PROVIDER inválido. Use 'google' ou 'ollama'."
            )

    def retrieve(self, query, top_k=None):
        """Recupera os chunks mais relevantes via busca híbrida com RRF.

        Parameters
        ----------
        query : str
            Pergunta do usuário utilizada para recuperar contexto.
        top_k : int, opcional
            Quantidade final de chunks retornados após a fusão dos rankings.

        Returns
        -------
        list[dict]
            Lista de chunks recuperados. Cada item contém `chunk_id`, `doc_id`,
            `texto` e `metadados`.
        """
        
        if top_k is None:
            top_k = self.default_top_k

        # 1. Recuperação Dense
        query_embedding = self.embedding_model.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 2 # Busca mais para fazer a fusão
        )
        dense_ids = dense_results['ids'][0]
        
        # 2. Recuperação Sparse (BM25)
        tokenized_query = _safe_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Pega os top K * 2 índices
        top_sparse_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        sparse_ids = [self.chunks_data[i]["chunk_id"] for i in top_sparse_idx]

        # 3. Fusão RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        k_rrf = self.rrf_k
        
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
            
        for rank, doc_id in enumerate(sparse_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)

        # Ordena e pega os K finais
        final_top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        # Monta a lista de contextos recuperados
        recovered_chunks = []
        for chunk in self.chunks_data:
            if chunk["chunk_id"] in final_top_ids:
                recovered_chunks.append(chunk)
                
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
            return "Não encontrei informações na base para responder a esta pergunta."
            
        context_str = ""
        for chunk in retrieved_chunks:
            context_str += f"\n\n--- INÍCIO DO TRECHO [{chunk['chunk_id']}] ---\n"
            context_str += f"Título: {chunk['metadados'].get('titulo', 'Documento sem título')}\n"
            context_str += f"Fonte: {chunk['metadados'].get('fonte', 'N/A')}\n"
            context_str += chunk['texto']
            context_str += f"\n--- FIM DO TRECHO [{chunk['chunk_id']}] ---"

        prompt_template = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template="""Você é um assistente tributário do Estado de Goiás.
Sua tarefa é responder à pergunta do utilizador APENAS com base no contexto fornecido.

REGRAS OBRIGATÓRIAS:
1. GROUNDING: Se a resposta não estiver no contexto, responda EXATAMENTE: "Não encontrei informações na base de conhecimento para responder a esta pergunta."
2. NÃO INVENTE: Não use conhecimentos externos à base fornecida. O foco não é aconselhamento jurídico genérico.
3. CITAÇÕES: Sempre que afirmar algo, cite a origem no final da frase com TÍTULO, LINK e CHUNK_ID.
   Formato exato: [Título do documento | URL da fonte | chunk_id]
   Exemplo: "O ICMS é isento neste caso [PARECER ECONOMIA/GEOT-15962 Nº 115/2022 | https://appasp.economia.go.gov.br/.../P_115_2022_SEI.docx | 77c6a4319534#chunk_0003]."
4. IDIOMA: Responda sempre em Português do Brasil (pt-BR), inclusive títulos, explicações e conclusões.
5. FORMATAÇÃO: Não use LaTeX, MathJax, Markdown matemático nem expressões como \text{{...}}. Quando houver fórmula, escreva em texto simples, por exemplo: "Crédito de ICMS = Valor total do combustível x alíquota de ICMS".
6. EVITE BLOCO DE FÓRMULA: Não escreva "Fórmula:" com notação simbólica. Prefira descrever o cálculo em frase clara usando "dividido por", "multiplicado por" e exemplos textuais.
7. CONCISÃO: Seja objetivo. Responda em no máximo 8 bullets curtos. Evite introduções longas, repetições e trechos enciclopédicos.
8. TABELAS: Não use tabelas em Markdown (com `|`). Quando precisar comparar conceitos, use lista de bullets com rótulos claros.

FORMATO OBRIGATÓRIO DE SAÍDA:
- Resposta direta: 1 a 2 bullets.
- Base legal: 2 a 4 bullets com referência normativa.
- Como aplicar: 2 a 3 bullets práticos.
- Citações: em todas as afirmações relevantes, no formato exigido.

CONTEXTO RECUPERADO:
{contexto}

PERGUNTA: {pergunta}
RESPOSTA:"""
        )
        
        prompt = prompt_template.format(contexto=context_str, pergunta=query)
        response = self.llm.invoke(prompt)
        answer = response.content
        answer = _normalize_math_notation(answer)
        answer = _normalize_markdown_tables(answer)
        return answer
