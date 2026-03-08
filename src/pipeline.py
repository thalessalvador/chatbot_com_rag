import argparse
import pandas as pd
import json
import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import nltk
from dotenv import load_dotenv
from tqdm import tqdm
try:
    from src.logging_config import get_logger
except ImportError:
    from logging_config import get_logger

nltk.download('punkt', quiet=True)
load_dotenv()
logger = get_logger(__name__)

# Configurações de Caminhos
RAW_DATA_PATH = "data/raw/rag_pareceres_resumo_geral_conclusao_202602131448.csv"
CHUNKS_PATH = "data/processed/chunks.json"
CHROMA_DB_DIR = "data/index/chroma_db"
BM25_INDEX_PATH = "data/index/bm25_index.pkl"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

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
    try:
        return int(os.getenv(var_name, str(default_value)))
    except ValueError:
        return default_value

def _get_bool_env(var_name, default_value):
    """Lê uma variável booleana de ambiente com fallback seguro.

    Parameters
    ----------
    var_name : str
        Nome da variável de ambiente.
    default_value : bool
        Valor padrão quando a variável não existe ou é inválida.

    Returns
    -------
    bool
        Valor booleano válido para uso na configuração.
    """
    raw = os.getenv(var_name)
    if raw is None:
        return default_value
    return raw.strip().lower() in {"1", "true", "yes", "on"}

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

def run_ingest():
    """Executa a etapa de ingestão e chunking dos documentos do corpus.

    A função lê o CSV bruto, segmenta o conteúdo textual em chunks e grava
    o resultado em JSON para uso nas etapas de indexação.

    Parameters
    ----------
    None
        Esta função não recebe parâmetros diretos; usa caminhos globais.

    Returns
    -------
    None
        Salva os chunks em arquivo e exibe logs no terminal.
    """
    logger.info("Iniciando Ingestão e Chunking...")
    
    if not os.path.exists(RAW_DATA_PATH):
        logger.error("Arquivo de dados não encontrado: %s", RAW_DATA_PATH)
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    chunk_size = _get_int_env("CHUNK_SIZE", 1000)
    chunk_overlap = _get_int_env("CHUNK_OVERLAP", 200)

    # Text Splitter configurável por ambiente para ajuste de contexto.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks = []
    
    for index, row in df.iterrows():
        doc_id = str(row.get('filename', f"doc_{index}")).replace('.docx', '').replace('.pdf', '')
        titulo = row.get('filename', 'Sem título')
        texto_completo = str(row.get('content', ''))
        
        # Gera os chunks
        chunks = text_splitter.split_text(texto_completo)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_linha{index}#chunk_{i:03d}"
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "texto": chunk_text,
                "metadados": {
                    "titulo": titulo,
                    "fonte": row.get('url_arquivo', 'Base Interna'),
                    "chunk_id": chunk_id
                }
            })
            
    # Salvar chunks processados
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    logger.info(
        "Ingestão concluída: %s chunks gerados e salvos em %s.",
        len(all_chunks),
        CHUNKS_PATH
    )

def run_index():
    """Executa a indexação híbrida Dense + Sparse (Trilha A).

    A função carrega os chunks processados, gera embeddings para indexação
    vetorial no ChromaDB e cria o índice BM25 para busca lexical.

    Parameters
    ----------
    None
        Esta função não recebe parâmetros diretos; usa caminhos globais.

    Returns
    -------
    None
        Persiste os índices no disco e exibe logs no terminal.
    """
    logger.info("Iniciando Indexação Híbrida...")
    
    if not os.path.exists(CHUNKS_PATH):
         logger.error("Chunks não encontrados. Rode a ingestão primeiro.")
         return
         
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # 1. Indexação Densa (ChromaDB)
    logger.info("Gerando embeddings dense e salvando no ChromaDB...")
    # Modelo de embeddings configurável por variável de ambiente.
    effective_embedding_device = _resolve_embedding_device(EMBEDDING_DEVICE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=effective_embedding_device)
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # Reseta coleção se existir para reindexação limpa
    try:
        chroma_client.delete_collection(name="pareceres_tributarios")
    except:
        pass
    collection = chroma_client.create_collection(name="pareceres_tributarios")

    textos = [c["texto"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadados = [c["metadados"] for c in chunks]
    
    # Gera embeddings em batch com barra de progresso visível no terminal.
    embedding_batch_size = _get_int_env("EMBEDDING_BATCH_SIZE", 64)
    show_progress = _get_bool_env("SHOW_PROGRESS", True)
    embeddings = model.encode(
        textos,
        batch_size=embedding_batch_size,
        show_progress_bar=show_progress
    ).tolist()

    # Grava no ChromaDB em lotes para mostrar progresso e reduzir sensação de travamento.
    chroma_add_batch_size = _get_int_env("CHROMA_ADD_BATCH_SIZE", 512)
    total_items = len(ids)
    for start in tqdm(
        range(0, total_items, chroma_add_batch_size),
        desc="Salvando no ChromaDB",
        disable=not show_progress
    ):
        end = min(start + chroma_add_batch_size, total_items)
        collection.add(
            embeddings=embeddings[start:end],
            documents=textos[start:end],
            metadatas=metadados[start:end],
            ids=ids[start:end]
        )

    # 2. Indexação Esparsa (BM25)
    logger.info("Gerando índice sparse (BM25)...")
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in textos]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Salva o modelo BM25 e o mapeamento de IDs
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)

    logger.info("Indexação concluída com sucesso.")

def run_evaluate():
    """Executa a etapa de avaliação do retriever (placeholder).

    No estado atual, a função apenas registra mensagens sobre a futura
    integração de cálculo de Recall@k com o conjunto dourado.

    Parameters
    ----------
    None
        Esta função não recebe parâmetros.

    Returns
    -------
    None
        Apenas imprime mensagens informativas.
    """
    logger.info("Executando avaliação do retriever no Golden Set...")
    # Aqui entraria a lógica de leitura do seu Feedback IA - editado.xlsx
    # e busca cruzada usando a função de retrieve do rag_core.py
    logger.info("Implementação de Recall@k pendente de conexão com Golden Set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG - Projeto Final")
    parser.add_argument('--step', choices=['ingest', 'index', 'evaluate'], required=True)
    args = parser.parse_args()

    if args.step == 'ingest':
        run_ingest()
    elif args.step == 'index':
        run_index()
    elif args.step == 'evaluate':
        run_evaluate()
