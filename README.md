# 📚 Projeto Final de PLN --- Chatbot Tributário com RAG Híbrido

Este repositório contém o **Projeto Final da disciplina de Processamento
de Linguagem Natural**, implementando um pipeline **RAG completo com a
Trilha A (Recuperação Híbrida: Sparse + Dense)**.

------------------------------------------------------------------------

## 📁 Estrutura do Projeto

``` text
project/
├── app/
│   └── app.py               # Interface Streamlit do chatbot
├── data/
│   ├── raw/                 # Coloque seu CSV original aqui
│   ├── processed/           # Chunks gerados são salvos aqui
│   └── index/               # Bancos ChromaDB e índice BM25
├── src/
│   ├── pipeline.py          # Scripts de Ingestão, Indexação e Avaliação
│   └── rag_core.py          # Lógica do Retriever Híbrido e LLM (Geração)
├── .env                     # Variáveis de ambiente (Chave da API)
├── Dockerfile               # Configuração da imagem Docker
├── docker-compose.yml       # Orquestração dos containers
└── requirements.txt         # Dependências do Python
```

------------------------------------------------------------------------

# 🚀 Como Executar (Totalmente Dockerizado)

## 1️⃣ Preparação

Crie um arquivo `.env` na raiz do projeto contendo sua chave da OpenAI:

``` env
OPENAI_API_KEY=sk-sua-chave-aqui
```

Certifique-se de que o arquivo CSV original:

`rag_pareceres_resumo_geral_conclusao_202602131448.csv`

está dentro da pasta:

`data/raw/`

------------------------------------------------------------------------

## 2️⃣ Passo a Passo da Execução

### 🔹 Passo A: Ingestão e Chunking

Processa o CSV, corta os textos e gera os IDs exigidos.

``` bash
docker-compose --profile setup up ingest
```

### 🔹 Passo B: Indexação Híbrida

Gera embeddings locais e índice BM25 para a Trilha A.

``` bash
docker-compose --profile setup up index
```

### 🔹 Passo C: Rodar o Chatbot (Streamlit)

``` bash
docker-compose --profile app up chatbot
```

Acesse no navegador:

http://localhost:8501

------------------------------------------------------------------------

# 🎯 Pontos Atendidos do Escopo Acadêmico

-   **Corpus com Metadados:** Extraídos do CSV (`doc_id`, `fonte`)
-   **Chunking:** Parametrizado no `pipeline.py` com tamanho fixo e
    overlap
-   **Citações e Grounding:** Engenharia de prompt estrita no
    `rag_core.py` exigindo `[doc_id#chunk_id]`
-   **Transparência:** Implementado via `expander` no Streamlit para
    visualizar o que foi recuperado
-   **Trilha A:** Uso de **Fusão de Ranks Recíprocos (RRF)** combinando:
    -   SentenceTransformers (Dense Retrieval)
    -   BM25Okapi (Sparse Retrieval)
