import streamlit as st
import sys
import os

# Adiciona o diretório raiz ao path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_core import HybridRAG

st.set_page_config(page_title="Chatbot Tributário - Projeto Final", page_icon="⚖️", layout="centered")

@st.cache_resource
def load_rag_system():
    try:
        return HybridRAG()
    except Exception as e:
        # Se falhar a carregar, regista o erro no terminal para podermos diagnosticar
        print(f"Erro Crítico ao carregar RAG: {e}")
        return None

st.title("⚖️ Chatbot de Pareceres Tributários")
st.markdown("""
**Projeto Final PLN - Trilha A (Recuperação Híbrida)**
Faça perguntas sobre ICMS, FUNDEINFRA e regimes especiais com base nos pareceres do SEI-GO.
""")

# Mostrar spinner enquanto os modelos (embeddings e llm) são carregados para a memória
with st.spinner("A inicializar os motores de Inteligência Artificial..."):
    rag = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta sobre tributação..."):
    # Exibe pergunta do utilizador
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera resposta
    with st.chat_message("assistant"):
        if rag is None:
            # Mensagem de erro amigável se os índices não existirem
            erro_msg = "⚠️ **Sistema não inicializado.** Os índices de busca não foram encontrados. Por favor, certifique-se de correr primeiro os comandos de Ingestão e Indexação."
            st.error(erro_msg)
            # Retira a pergunta anterior do histórico já que não conseguimos responder
            if st.session_state.messages:
                st.session_state.messages.pop() 
        else:
            with st.spinner("Buscando nos pareceres..."):
                try:
                    # 1. Retrieval
                    retrieved_chunks = rag.retrieve(prompt, top_k=5)
                    
                    # 2. Generation
                    answer = rag.generate_answer(prompt, retrieved_chunks)
                    
                    # 3. Exibição da resposta
                    st.markdown(answer)
                    
                    # Salva no histórico APENAS se houver uma resposta válida
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Requisito: Transparência (Mostrar trechos recuperados)
                    with st.expander("🔍 Ver Trechos Recuperados (Transparência)"):
                        if retrieved_chunks:
                            for idx, chunk in enumerate(retrieved_chunks):
                                st.markdown(f"**Trecho {idx+1} (ID: `{chunk['chunk_id']}`)**")
                                st.markdown(f"**Fonte:** {chunk['metadados']['fonte']}")
                                st.info(chunk['texto'])
                        else:
                            st.write("Nenhum trecho relevante encontrado.")
                            
                except Exception as e:
                    # Captura qualquer erro (ex: Chave de API inválida, erro de rede)
                    st.error(f"❌ **Ocorreu um erro ao processar o pedido:** {str(e)}")
                    # Removemos a pergunta do user do histórico para não encravar a conversa
                    if st.session_state.messages:
                        st.session_state.messages.pop()