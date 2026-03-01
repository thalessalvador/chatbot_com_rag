import json
import pickle
import chromadb
import os
from sentence_transformers import SentenceTransformer
import nltk

# FIX: Garante que os pacotes de tokenização do NLTK são descarregados no contentor do chatbot
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

class HybridRAG:
    def __init__(self, chroma_path="data/index/chroma_db", bm25_path="data/index/bm25_index.pkl"):
        # Dense
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(name="pareceres_tributarios")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sparse
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks_data = data['chunks']
            
        # LLM (Usando Google Gemini via Langchain - Requer GOOGLE_API_KEY no .env)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def retrieve(self, query, top_k=5):
        """TRILHA A: Recuperação Híbrida (Sparse + Dense) usando RRF"""
        
        # 1. Recuperação Dense
        query_embedding = self.embedding_model.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 2 # Busca mais para fazer a fusão
        )
        dense_ids = dense_results['ids'][0]
        
        # 2. Recuperação Sparse (BM25)
        tokenized_query = nltk.word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Pega os top K * 2 índices
        top_sparse_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        sparse_ids = [self.chunks_data[i]["chunk_id"] for i in top_sparse_idx]

        # 3. Fusão RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        k_rrf = 60 # Constante padrão RRF
        
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
        """Geração com Grounding, Recusa e Citações rigorosas."""
        
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