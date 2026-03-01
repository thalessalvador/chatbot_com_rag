import argparse
import pandas as pd
import json
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import nltk

nltk.download('punkt', quiet=True)

# Configurações de Caminhos
RAW_DATA_PATH = "data/raw/rag_pareceres_resumo_geral_conclusao_202602131448.csv"
CHUNKS_PATH = "data/processed/chunks.json"
CHROMA_DB_DIR = "data/index/chroma_db"
BM25_INDEX_PATH = "data/index/bm25_index.pkl"

def run_ingest():
    """Etapa 1: Ingestão e Chunking"""
    print("Iniciando Ingestão e Chunking...")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Erro: Arquivo {RAW_DATA_PATH} não encontrado.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    # Text Splitter: 1000 caracteres, 200 de overlap para manter contexto tributário
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
        
    print(f"Ingestão concluída! {len(all_chunks)} chunks gerados e salvos em {CHUNKS_PATH}.")

def run_index():
    """Etapa 2: Indexação Híbrida (Dense + Sparse) - TRILHA A"""
    print("Iniciando Indexação Híbrida...")
    
    if not os.path.exists(CHUNKS_PATH):
         print("Erro: Chunks não encontrados. Rode a ingestão primeiro.")
         return
         
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # 1. Indexação Densa (ChromaDB)
    print("Gerando Embeddings Dense e salvando no ChromaDB...")
    # Usando modelo open-source leve para rodar em CPU sem custo
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
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
    
    # Gera embeddings em batch (pode demorar um pouco na CPU)
    embeddings = model.encode(textos).tolist()
    
    collection.add(
        embeddings=embeddings,
        documents=textos,
        metadatas=metadados,
        ids=ids
    )

    # 2. Indexação Esparsa (BM25)
    print("Gerando Índice Sparse (BM25)...")
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in textos]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Salva o modelo BM25 e o mapeamento de IDs
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)

    print("Indexação concluída com sucesso!")

def run_evaluate():
    """Etapa 4: Avaliação Básica (Recall@k)"""
    print("Executando Avaliação do Retriever no Golden Set...")
    # Aqui entraria a lógica de leitura do seu Feedback IA - editado.xlsx
    # e busca cruzada usando a função de retrieve do rag_core.py
    print("Implementação de Recall@k pendente de conexão com Golden Set.")

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