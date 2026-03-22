import pandas as pd
import os
import urllib.parse

# Importa a classe do seu projeto que faz as buscas
from rag_core import HybridRAG

def extrair_nome_arquivo(caminho):
    """
    Extrai apenas o nome do ficheiro (ex: P_104_2022_SEI.docx) de um caminho completo.
    Exemplo: https://.../.../P_104_2022_SEI.docx -> P_104_2022_SEI.docx
    """
    if pd.isna(caminho) or caminho == "Nenhum" or caminho == "Erro":
        return ""
    # Pega tudo o que está depois da última barra '/'
    nome = str(caminho).split('/')[-1]
    # Limpa caracteres especiais de URLs (%20 vira espaço, etc)
    return urllib.parse.unquote(nome)

def calcular_recall(golden_file, k=5):
    """
    Testa o sistema RAG iterando sobre as perguntas do Golden Set e comparando
    o documento retornado com o documento esperado (Gabarito).
    """
    print("A inicializar o RAG para Avaliação (Aguarde o carregamento dos modelos)...")
    
    # Instancia o seu sistema RAG apontando para os índices corretos
    # (Removido os ../ pois o script será rodado a partir da raiz do projeto)
    rag = HybridRAG(
        chroma_path="data/index/chroma_db", 
        bm25_path="data/index/bm25_index.pkl"
    )
    
    # Carrega o Golden Set
    df = pd.read_excel(golden_file)
    
    # Modos que vamos testar
    modos = ['dense', 'sparse', 'hybrid']
    
    # Dicionário para guardar os acertos e totais de cada modo
    resultados = {modo: {'acertos': 0, 'total_validos': 0} for modo in modos}
    
    print(f"\nA avaliar Recall@{k} para {len(df)} perguntas...\n")
    
    for idx, row in df.iterrows():
        pergunta = row['Pergunta']
        doc_esperado_bruto = row['documento_esperado']
        
        # Extrai o nome limpo do documento que o sistema DEVERIA encontrar
        doc_esperado = extrair_nome_arquivo(doc_esperado_bruto)
        
        # Ignora linhas em que não temos um gabarito definido
        if not doc_esperado:
            continue
            
        for modo in modos:
            resultados[modo]['total_validos'] += 1
            
            # Faz a busca usando o modo atual (denso, esparso ou híbrido)
            chunks_recuperados = rag.retrieve(pergunta, top_k=k, mode=modo)
            
            # Verifica se o documento esperado está dentro do Top K retornado
            acertou = False
            for chunk in chunks_recuperados:
                # O documento pode estar na 'fonte' ou no 'chunk_id' dentro dos metadados
                fonte_chunk = extrair_nome_arquivo(chunk.get('metadados', {}).get('fonte', ''))
                chunk_id = chunk.get('chunk_id', '')
                
                if doc_esperado in fonte_chunk or doc_esperado in chunk_id:
                    acertou = True
                    break
            
            # Conta o acerto se encontrou
            if acertou:
                resultados[modo]['acertos'] += 1
                
    # Imprime os resultados finais
    print("-" * 40)
    print(f"📊 RESULTADOS DA AVALIAÇÃO (PR5) - Recall@{k} 📊")
    print("-" * 40)
    for modo in modos:
        acertos = resultados[modo]['acertos']
        total = resultados[modo]['total_validos']
        recall = (acertos / total) * 100 if total > 0 else 0
        
        print(f"Modo {modo.upper():<7} -> Recall@{k}: {recall:.1f}% ({acertos}/{total} acertos)")
        
if __name__ == "__main__":
    # Caminho ajustado para rodar a partir da raiz do projeto
    golden_path = "Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx" 
    
    if os.path.exists(golden_path):
        calcular_recall(golden_path, k=5)
    else:
        print(f"ERRO: Ficheiro {golden_path} não encontrado!")
        print("Certifique-se de que o ficheiro está na pasta raiz do projeto.")