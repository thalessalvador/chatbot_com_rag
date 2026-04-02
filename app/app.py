import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_core import HybridRAG
from src.app_config import get_config_value
from src.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)
NO_CONTEXT_RESPONSE = str(
    get_config_value("rag.no_context_response")
)


def _is_no_context_response(answer_text):
    """Indica se a resposta corresponde ao fallback de ausência de contexto."""
    if not answer_text:
        return False
    return answer_text.strip().strip('"').strip() == NO_CONTEXT_RESPONSE


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
        "RETRIEVAL_TOP_K": "retrieval.top_k",
    }
    try:
        return int(get_config_value(mapping.get(var_name, ""), default_value))
    except (TypeError, ValueError):
        return default_value


st.set_page_config(page_title="Chatbot Tributário - Projeto Final", page_icon="⚖️", layout="centered")


@st.cache_resource
def load_rag_system(llm_provider, llm_model, ollama_base_url, embedding_model_id):
    """Carrega e cacheia a instância principal do sistema RAG.

    `embedding_model_id` entra na chave de cache para que, após reindexar com
    outro modelo em `config.yaml`, não se reuse o SentenceTransformer antigo
    sobre vetores novos no ChromaDB.

    Parameters
    ----------
    llm_provider : str
        Provedor de LLM selecionado (`google` ou `ollama`).
    llm_model : str
        Nome do modelo do provedor escolhido.
    ollama_base_url : str
        URL base da API do Ollama (utilizada quando o provedor é `ollama`).
    embedding_model_id : str
        Valor de `embeddings.model` no momento da carga (identificador de cache).

    Returns
    -------
    HybridRAG | None
        Instância pronta para consulta ou `None` em caso de falha de carga.
    """
    try:
        return HybridRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
        )
    except Exception as e:
        logger.exception("Erro crítico ao carregar o sistema RAG: %s", e)
        return None


st.title("⚖️ Chatbot de Pareceres Tributários")
st.markdown(
    """
**Projeto Final PLN - Trilha A (Recuperação Híbrida)**
Faça perguntas sobre ICMS, FUNDEINFRA e regimes especiais com base nos pareceres do SEI-GO.

⚠️ *Aviso: As respostas deste chatbot têm caráter apenas explicativo com base no corpus. Procure orientação profissional para aconselhamento jurídico.*
"""
)

st.sidebar.header("Configuração do LLM")
llm_provider = st.sidebar.selectbox(
    "Provedor",
    options=["google", "ollama"],
    index=0 if str(get_config_value("llm.provider")).lower() == "google" else 1,
)

default_model = (
    get_config_value("llm.google_model")
    if llm_provider == "google"
    else get_config_value("llm.ollama_model")
)
llm_model = st.sidebar.text_input("Modelo", value=default_model)
ollama_base_url = st.sidebar.text_input(
    "Ollama Base URL",
    value=get_config_value("llm.ollama_base_url"),
    disabled=llm_provider != "ollama",
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuração de Pesquisa")
busca_selecionada = st.sidebar.radio(
    "Estratégia:",
    options=["Híbrido (Denso + Esparso)", "Denso (Apenas Embeddings)", "Esparso (Apenas BM25)"],
    index=0,
    help="Define como a IA vai procurar os documentos na base de dados."
)

mapa_modos = {
    "Híbrido (Denso + Esparso)": "hybrid",
    "Denso (Apenas Embeddings)": "dense",
    "Esparso (Apenas BM25)": "sparse"
}
modo_escolhido = mapa_modos[busca_selecionada]

with st.spinner("A inicializar os motores de Inteligência Artificial..."):
    rag = load_rag_system(
        llm_provider,
        llm_model,
        ollama_base_url,
        str(get_config_value("embeddings.model", "")),
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta sobre tributação..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if rag is None:
            erro_msg = (
                "⚠️ **Sistema não inicializado.** Os índices de busca não foram encontrados. "
                "Por favor, certifique-se de correr primeiro os comandos de Ingestão e Indexação."
            )
            st.error(erro_msg)
            if st.session_state.messages:
                st.session_state.messages.pop()
        else:
            try:
                retrieval_top_k = _get_int_env("RETRIEVAL_TOP_K", 5)
                with st.spinner("Recuperando trechos nos pareceres…"):
                    retrieved_chunks = rag.retrieve(
                        prompt, top_k=retrieval_top_k, mode=modo_escolhido
                    )
                _llm_hint = (
                    "Gerando resposta com o LLM"
                    + (" (Ollama local: pode levar 1–3 min com prompt grande)…"
                       if llm_provider == "ollama"
                       else "…")
                )
                with st.spinner(_llm_hint):
                    answer = rag.generate_answer(prompt, retrieved_chunks)
                st.markdown(answer, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if not _is_no_context_response(answer):
                    with st.expander("🔍 Ver Trechos Recuperados (Transparência)"):
                        if retrieved_chunks:
                            for idx, chunk in enumerate(retrieved_chunks):
                                metadados = chunk.get("metadados", {})
                                titulo = metadados.get("titulo", "Documento sem título")
                                fonte = metadados.get("fonte", "")
                                chunk_id = chunk.get("chunk_id", "N/A")

                                st.markdown(f"**Trecho {idx+1}**")
                                if fonte:
                                    st.markdown(
                                        (
                                            f"<strong>Documento:</strong> "
                                            f"<a href=\"{fonte}\" target=\"_blank\" rel=\"noopener noreferrer\">"
                                            f"{titulo} ↗</a> "
                                            f"| <strong>ID:</strong> <code>{chunk_id}</code>"
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(f"**Documento:** {titulo} | **ID:** `{chunk_id}`")
                                st.info(chunk["texto"])
                        else:
                            st.write("Nenhum trecho relevante encontrado.")

            except Exception as e:
                error_text = str(e)
                logger.exception("Erro ao processar pergunta: %s", error_text)
                st.error(f"❌ **Ocorreu um erro ao processar o pedido:** {error_text}")
                if st.session_state.messages:
                    st.session_state.messages.pop()
