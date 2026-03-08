import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_core import HybridRAG
from src.logging_config import get_logger

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
    try:
        return int(os.getenv(var_name, str(default_value)))
    except ValueError:
        return default_value

st.set_page_config(page_title="Chatbot Tributário - Projeto Final", page_icon="⚖️", layout="centered")

@st.cache_resource
def load_rag_system(llm_provider, llm_model, ollama_base_url):
    """Carrega e cacheia a instância principal do sistema RAG.

    Parameters
    ----------
    llm_provider : str
        Provedor de LLM selecionado (`google` ou `ollama`).
    llm_model : str
        Nome do modelo do provedor escolhido.
    ollama_base_url : str
        URL base da API do Ollama (utilizada quando o provedor é `ollama`).

    Returns
    -------
    HybridRAG | None
        Instância pronta para consulta ou `None` em caso de falha de carga.
    """
    try:
        return HybridRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url
        )
    except Exception as e:
        # Se falhar a carregar, regista o erro no terminal para podermos diagnosticar
        logger.exception("Erro crítico ao carregar o sistema RAG: %s", e)
        return None

st.title("⚖️ Chatbot de Pareceres Tributários")
st.markdown("""
**Projeto Final PLN - Trilha A (Recuperação Híbrida)**
Faça perguntas sobre ICMS, FUNDEINFRA e regimes especiais com base nos pareceres do SEI-GO.
""")

st.sidebar.header("Configuração do LLM")
llm_provider = st.sidebar.selectbox(
    "Provedor",
    options=["google", "ollama"],
    index=0 if os.getenv("LLM_PROVIDER", "google").lower() == "google" else 1
)

default_model = (
    os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if llm_provider == "google"
    else os.getenv("OLLAMA_MODEL", "llama3.1:8b")
)
llm_model = st.sidebar.text_input("Modelo", value=default_model)
ollama_base_url = st.sidebar.text_input(
    "Ollama Base URL",
    value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    disabled=llm_provider != "ollama"
)

# Mostrar spinner enquanto os modelos (embeddings e llm) são carregados para a memória
with st.spinner("A inicializar os motores de Inteligência Artificial..."):
    rag = load_rag_system(llm_provider, llm_model, ollama_base_url)

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
                    retrieval_top_k = _get_int_env("RETRIEVAL_TOP_K", 5)
                    retrieved_chunks = rag.retrieve(prompt, top_k=retrieval_top_k)
                    
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
