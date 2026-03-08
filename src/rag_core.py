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
            context_str += f"Fonte: {chunk['metadados']['fonte']}\n"
            context_str += chunk['texto']
            context_str += f"\n--- FIM DO TRECHO [{chunk['chunk_id']}] ---"

        prompt_template = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template="""Você é um assistente tributário do Estado de Goiás.
Sua tarefa é responder à pergunta do utilizador APENAS com base no contexto fornecido.

REGRAS OBRIGATÓRIAS:
1. GROUNDING: Se a resposta não estiver no contexto, responda EXATAMENTE: "Não encontrei informações na base de conhecimento para responder a esta pergunta."
2. NÃO INVENTE: Não use conhecimentos externos à base fornecida. O foco não é aconselhamento jurídico genérico.
3. CITAÇÕES: Sempre que afirmar algo, deve citar a origem da informação usando o formato exato [doc_id#chunk_id] no final da frase. Exemplo: "O ICMS é isento neste caso [P_239_2023_SEI#chunk_002]."

CONTEXTO RECUPERADO:
{contexto}

PERGUNTA: {pergunta}
RESPOSTA:"""
        )
        
        prompt = prompt_template.format(contexto=context_str, pergunta=query)
        response = self.llm.invoke(prompt)
        
        return response.content
