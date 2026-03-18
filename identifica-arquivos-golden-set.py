import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 1. Carrega o seu Golden Set
caminho_excel = 'Feedback IA_editado.xlsx'
df_golden = pd.read_excel(caminho_excel)
documentos_esperados = []

# 2. Inicializa o ChromaDB
print("A inicializar a base de vetores (ChromaDB)...")
chroma_client = chromadb.PersistentClient(
    path="data/index/chroma_db", 
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection(name="pareceres_tributarios")

# CORREÇÃO AQUI: Mudamos para o modelo que gera 384 dimensões (padrão da sua base)
print("A carregar modelo de embeddings (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nA procurar os documentos na base Vetorial...")

for idx, row in df_golden.iterrows():
    texto_para_buscar = str(row['Resposta esperada (com base no parecer)'])
    
    try:
        # Codifica a resposta esperada em vetor (agora com 384 dimensões)
        query_embedding = embedding_model.encode([texto_para_buscar]).tolist()
        
        # Busca o chunk mais parecido no ChromaDB (Top 1)
        resultados = collection.query(
            query_embeddings=query_embedding,
            n_results=1
        )
        
        # Verifica se obteve resultados válidos
        if resultados and resultados['metadatas'] and len(resultados['metadatas'][0]) > 0:
            metadados = resultados['metadatas'][0][0]
            
            # Tenta pegar a 'fonte' ou o 'titulo' do documento a partir dos metadados
            doc_id = metadados.get('fonte', metadados.get('titulo', 'N/A'))
            documentos_esperados.append(doc_id)
        else:
            documentos_esperados.append("Nenhum")
            
    except Exception as e:
        print(f"Erro na linha {idx}: {e}")
        documentos_esperados.append("Erro")

# 3. Adiciona a coluna e guarda o ficheiro final
df_golden['documento_esperado'] = documentos_esperados
ficheiro_saida = 'Golden_Set_Preenchido_pelo_RAG.xlsx'
df_golden.to_excel(ficheiro_saida, index=False)

print(f"\nFinalizado! O ficheiro foi guardado como: {ficheiro_saida}")