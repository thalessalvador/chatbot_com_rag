import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import os
import time
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# --- CONFIGURAÇÕES DO PROXY ---
PROXY_BASE_URL = 'https://gateway-ia.economia.go.gov.br'
PROXY_GEMINI_USER_ID = "DEVCARLOS-m9u8rp9jn7O3XR4-"
PROXY_MODEL_PATH = "v1beta/models/gemini-2.0-flash:generateContent"
# ------------------------------

caminho_excel = 'Feedback IA_editado.xlsx'
df_golden = pd.read_excel(caminho_excel)
documentos_esperados = []

print("A inicializar a base de vetores (ChromaDB)...")
chroma_client = chromadb.PersistentClient(
    path="data/index/chroma_db", 
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection(name="pareceres_tributarios")

print("A carregar modelo de embeddings (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nA recuperar top-5 com o ChromaDB e a validar o Gabarito via Proxy...\n")

for idx, row in df_golden.iterrows():
    pergunta = str(row['Pergunta'])
    resposta_esperada = str(row['Resposta esperada (com base no parecer)'])
    
    print(f"\n{'-'*60}")
    print(f"[{idx+1}/{len(df_golden)}] AVALIANDO PERGUNTA: {pergunta}")
    
    # IMPORTANTE: Usamos a Resposta Esperada para forçar o banco a trazer o arquivo correto
    texto_para_buscar = f"{pergunta} {resposta_esperada}"
    
    try:
        query_embedding = embedding_model.encode([texto_para_buscar]).tolist()
        resultados = collection.query(
            query_embeddings=query_embedding,
            n_results=100
        )
        
        if resultados and resultados['metadatas'] and len(resultados['metadatas'][0]) > 0:
            metadados_lista = resultados['metadatas'][0]
            textos_lista = resultados['documents'][0]
            
            nomes_documentos = []
            for meta in metadados_lista:
                # Prioriza a chave 'fonte' (onde fica armazenada a URL original como https://appasp...)
                doc_id = meta.get('fonte', meta.get('titulo', 'Doc_Desconhecido'))
                nomes_documentos.append(doc_id)
            
            print(f"[{idx+1}/{len(df_golden)}] ChromaDB recuperou os seguintes arquivos:")
            for i, nome in enumerate(nomes_documentos):
                print(f"  {i+1}. {nome}")
            
            # Prompt focado em encontrar a FONTE EXATA baseada na resposta esperada
            prompt = "Você é um auditor responsável por mapear o documento de origem de uma resposta jurídica.\n"
            prompt += "Abaixo está uma PERGUNTA, a RESPOSTA ESPERADA (Gabarito) e 5 trechos de documentos.\n\n"
            prompt += f"PERGUNTA: {pergunta}\n"
            prompt += f"RESPOSTA ESPERADA: {resposta_esperada}\n\n"
            prompt += "TRECHOS RECUPERADOS:\n"
            
            for i, (meta, texto) in enumerate(zip(metadados_lista, textos_lista)):
                doc_id = meta.get('fonte', meta.get('titulo', f'Doc_Desconhecido_{i}'))
                prompt += f"--- OPÇÃO {i+1} (Fonte: {doc_id}) ---\n{texto}\n\n"
                
            prompt += "TAREFA:\n"
            prompt += "1. Identifique qual das OPÇÕES contém o texto que melhor fundamenta a RESPOSTA ESPERADA.\n"
            prompt += "2. A sua ÚLTIMA LINHA de resposta deve obrigatoriamente copiar e colar de forma exata o link/nome listado em 'Fonte' da opção vencedora.\n"
            prompt += "Exemplo do formato exigido:\n"
            prompt += "MELHOR DOCUMENTO: https://appasp.economia.go.gov.br/pareceres/arquivos/Fundeinfra/P_82_2023_SEI.docx\n"
            prompt += "Se nenhum trecho for útil, responda: MELHOR DOCUMENTO: Nenhum"

            url = f"{PROXY_BASE_URL}/gemini-proxy/{PROXY_MODEL_PATH}"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': PROXY_GEMINI_USER_ID,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "safetySettings": safety_settings,
                "generationConfig": {
                    "temperature": 0.0, 
                    "maxOutputTokens": 300 
                }
            }
            
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                verify=False,
                timeout=300
            )
            
            if response.status_code == 200:
                resposta_gemini = response.json()
                candidato = resposta_gemini.get('candidates', [{}])[0]
                
                if 'content' in candidato and 'parts' in candidato['content']:
                    texto_resposta = candidato['content']['parts'][0]['text'].strip()
                    
                    linhas = texto_resposta.split('\n')
                    melhor_doc = "Nenhum"
                    
                    for linha in reversed(linhas):
                        if "MELHOR DOCUMENTO:" in linha.upper():
                            melhor_doc = linha.split(":", 1)[1].strip()
                            melhor_doc = melhor_doc.replace('*', '').replace('[', '').replace(']', '')
                            break
                            
                    print(f"[{idx+1}/{len(df_golden)}] LLM Escolheu a Fonte: {melhor_doc}")
                    documentos_esperados.append(melhor_doc)
                else:
                    motivo = candidato.get('finishReason', 'Desconhecido')
                    print(f"[{idx+1}/{len(df_golden)}] Resposta da API bloqueada/vazia. Motivo: {motivo}")
                    documentos_esperados.append(f"Erro: {motivo}")
            else:
                print(f"[{idx+1}/{len(df_golden)}] Erro HTTP da API {response.status_code}")
                documentos_esperados.append("Erro API")
            
            time.sleep(1)
            
        else:
            print(f"[{idx+1}/{len(df_golden)}] Nenhum documento recuperado pelo ChromaDB.")
            documentos_esperados.append("Nenhum")
            
    except requests.exceptions.RequestException as req_err:
        print(f"[{idx+1}/{len(df_golden)}] Erro de conexão com o proxy: {req_err}")
        documentos_esperados.append("Erro Conexão Proxy")
    except Exception as e:
        print(f"[{idx+1}/{len(df_golden)}] Erro inesperado ao processar a linha {idx}: {e}")
        documentos_esperados.append("Erro")

df_golden['documento_esperado'] = documentos_esperados
ficheiro_saida = 'Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx'
df_golden.to_excel(ficheiro_saida, index=False)

print(f"\n{'-'*60}")
print(f"Finalizado! O ficheiro com o gabarito gerado foi guardado como: {ficheiro_saida}")