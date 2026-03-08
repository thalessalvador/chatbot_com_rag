# Projeto Final PLN - Chatbot Tributário com RAG Híbrido

Este projeto implementa um pipeline de **RAG híbrido** para consulta de pareceres:
- recuperação densa com `SentenceTransformers` + `ChromaDB`;
- recuperação esparsa com `BM25`;
- fusão de rankings com `RRF (Reciprocal Rank Fusion)`;
- interface web com `Streamlit`.

## Requisito de versão do Python

Este projeto deve ser executado com **Python 3.11**.

- Versões como Python 3.12 podem falhar na instalação de dependências nativas (ex.: `chroma-hnswlib`).
- Recomendação: criar o ambiente virtual explicitamente com `py -3.11 -m venv .venv`.

## Estrutura do projeto

```text
app/app.py                 # Interface Streamlit
src/pipeline.py            # Ingestão e indexação
src/rag_core.py            # Retriever híbrido + geração da resposta
data/raw/                  # Entrada de documentos (CSV)
data/processed/chunks.json # Saída da ingestão
data/index/                # Índices ChromaDB e BM25
requirements.txt           # Dependências Python
.env                       # Configuração de provedores e modelos
```

## Como o fluxo funciona

1. Ingestão (`python src/pipeline.py --step ingest`)
- Lê o CSV bruto.
- Gera chunks de texto (`chunk_size=1000`, `chunk_overlap=200`).
- Salva em `data/processed/chunks.json`.

2. Indexação (`python src/pipeline.py --step index`)
- Gera embeddings com modelo configurado em `EMBEDDING_MODEL` no `.env`.
- Cria índice vetorial no ChromaDB.
- Cria índice BM25 para busca lexical.

3. Consulta (`streamlit run app/app.py`)
- Recupera candidatos dense + sparse.
- Aplica RRF para ranking final.
- Envia contexto ao LLM.
- Exibe resposta e os trechos recuperados.

## Entrada de documentos

O projeto atual usa CSV em `data/raw/`.

Arquivo padrão esperado:
- `data/raw/rag_pareceres_resumo_geral_conclusao_202602131448.csv`

Colunas utilizadas no pipeline:
- `filename`: nome/título do documento;
- `content`: texto completo;
- `url_arquivo`: referência/fonte para metadados.

Se quiser outro arquivo, altere `RAW_DATA_PATH` em `src/pipeline.py`.

## Configuração (.env)

Exemplo recomendado para uso local com Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=ministral-3:14b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.0

# Embeddings (retrieval)
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DEVICE=cuda
RETRIEVAL_TOP_K=5
RRF_K=60
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SHOW_PROGRESS=true
EMBEDDING_BATCH_SIZE=64
CHROMA_ADD_BATCH_SIZE=512

# Para usar Google/Gemini, descomente e troque o provedor:
# LLM_PROVIDER=google
# GOOGLE_API_KEY=sua_chave_google_aqui
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0.0
```

Ao trocar `EMBEDDING_MODEL`, rode novamente:
- `python src/pipeline.py --step index`

## Logging (arquivo rotativo + console)

O projeto usa logging centralizado com saída no terminal e em arquivo rotativo.

Configuração no `.env`:
- `LOG_LEVEL` (padrão: `INFO`)
- `LOG_FILE` (padrão: `logs/app.log`)
- `LOG_MAX_BYTES` (padrão: `5242880`)
- `LOG_BACKUP_COUNT` (padrão: `5`)

Formato das linhas:
- `%(asctime)s | %(levelname)s | %(name)s | %(message)s`

Exemplo de troubleshooting rápido:
- acompanhar logs em tempo real:

```powershell
Get-Content logs/app.log -Wait
```

- validar rotação com limite pequeno:
1. ajustar `LOG_MAX_BYTES=10240` no `.env`
2. executar pipeline/app
3. verificar se surgem arquivos `app.log.1`, `app.log.2`, etc.

## Ajustes de performance no Ollama

No `.env`, você pode ajustar:
- `OLLAMA_NUM_CTX`: tamanho da janela de contexto (menor tende a ser mais rápido).
- `OLLAMA_NUM_PREDICT`: limite de tokens de saída.
- `OLLAMA_NUM_GPU`: nível de offload para GPU.
- `OLLAMA_NUM_THREAD`: número de threads CPU (`0` = automático).
- `OLLAMA_KEEP_ALIVE`: mantém o modelo carregado para evitar recarga frequente.
- `OLLAMA_TEMPERATURE`: controla aleatoriedade da geração (`0.0` mais determinístico).


## Parâmetros de recuperação e chunking

Também são configuráveis no `.env`:
- `RETRIEVAL_TOP_K`: quantidade de chunks finais enviados ao prompt.
- `RRF_K`: constante da fusão de ranking RRF.
- `CHUNK_SIZE`: tamanho dos chunks em caracteres.
- `CHUNK_OVERLAP`: sobreposição entre chunks.
- `SHOW_PROGRESS`: exibe barras de progresso na indexação.
- `EMBEDDING_BATCH_SIZE`: tamanho de lote para gerar embeddings.
- `CHROMA_ADD_BATCH_SIZE`: tamanho de lote para gravar no ChromaDB.


Quando alterar `CHUNK_SIZE` ou `CHUNK_OVERLAP`, rode novamente:
- `python src/pipeline.py --step ingest`
- `python src/pipeline.py --step index`

Quando alterar `EMBEDDING_MODEL`, rode novamente:
- `python src/pipeline.py --step index`

## Instalação do Ollama (Windows)

1. Baixe e instale o Ollama:
- Site oficial: `https://ollama.com/download/windows`

2. Verifique se o Ollama está disponível no terminal:

```powershell
ollama --version
```

3. Baixe o modelo local que será usado no projeto (exemplo atual):

```powershell
ollama pull ministral-3:14b
```

4. Teste o modelo:

```powershell
ollama run ministral-3:14b
```

5. Confirme os modelos instalados:

```powershell
ollama list
```

6. Configure o `.env`:
- Execução local (Streamlit fora do Docker): `OLLAMA_BASE_URL=http://localhost:11434`
- Execução via Docker (app no container): `OLLAMA_BASE_URL=http://host.docker.internal:11434`

## Setup local (Windows / PowerShell)

1. Criar e ativar ambiente virtual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip setuptools
```

2. Instalar dependências:

```powershell
pip install -r requirements.txt
```

3. Rodar pipeline e app:

```powershell
python src/pipeline.py --step ingest
python src/pipeline.py --step index
streamlit run app/app.py
```

4. Acessar:
- `http://localhost:8501`

## Execução do zero (Windows / PowerShell)

Sequência completa para subir o projeto do zero até abrir o chatbot:

1. Ir para a pasta do projeto:

```powershell
cd C:\Projetos\Pos_ia\PLN\chatbot_com_rag
```

2. Criar e ativar ambiente virtual com Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

3. Instalar dependências:

```powershell
python -m pip install --upgrade pip setuptools
pip install -r requirements.txt
```

4. Garantir que o Ollama está instalado e com o modelo:

```powershell
ollama --version
ollama pull ministral-3:14b
ollama list
```

5. Configurar `.env` (mínimo recomendado):

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=ministral-3:14b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DEVICE=cuda
```

6. Executar ingestão:

```powershell
python src/pipeline.py --step ingest
```

7. Executar indexação:

```powershell
python src/pipeline.py --step index
```

8. Abrir o chatbot:

```powershell
streamlit run app/app.py
```

9. Acessar no navegador:
- `http://localhost:8501`

## Configuração de GPU (CUDA) vs CPU

Este projeto roda em CPU e GPU para a etapa de embeddings.

Configuração principal no `.env`:
- `EMBEDDING_DEVICE=cpu` para compatibilidade imediata.
- `EMBEDDING_DEVICE=cuda` para tentar usar GPU.

Verificação rápida:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Se retornar `False`/`no-cuda`, você está com PyTorch sem suporte CUDA.

Instalação PyTorch com CUDA (exemplo cu121):

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Observação:
- ajuste `cu121` para a versão necessária no seu ambiente (`cu118`, `cu124`, etc.).

Troubleshooting para GPUs novas (ex.: RTX série 50), se necessário:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Comportamento do projeto:
- Se `EMBEDDING_DEVICE=cuda` e CUDA não estiver disponível, o código faz fallback automático para `cpu` e registra aviso no terminal.

## Setup com Docker

1. Ajustar `.env`.

Se Ollama estiver no host e o app no container, use:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

2. Ingestão:

```bash
docker-compose --profile setup up ingest
```

3. Indexação:

```bash
docker-compose --profile setup up index
```

4. Chatbot:

```bash
docker-compose --profile app up chatbot
```

## Provedores de LLM suportados

`LLM_PROVIDER=ollama`:
- usa `langchain-ollama`;
- modelo definido por `OLLAMA_MODEL`;
- endpoint definido por `OLLAMA_BASE_URL`.

`LLM_PROVIDER=google`:
- requer instalação adicional:

```powershell
pip install -r requirements-google.txt
```

- requer `GOOGLE_API_KEY` no `.env`;
- modelo definido por `GOOGLE_MODEL`.

Se o ambiente já tiver ficado inconsistente (ex.: `pip check` com conflitos em
`protobuf`, `packaging` e `tenacity`), rode:

```powershell
pip install --upgrade --force-reinstall `
  "packaging<24" `
  "protobuf>=3.20,<5" `
  "tenacity>=8.1.0,<9" `
  "langchain-core>=0.3.75,<0.4.0" `
  "langchain-ollama==0.2.3" `
  "langchain-text-splitters==0.3.6" `
  "langchain-google-genai==2.1.12"
```

Depois valide:

```powershell
pip check
```

## Dependências principais

Definidas em `requirements.txt`:
- `streamlit`
- `pandas`, `numpy`
- `chromadb`, `sentence-transformers`
- `rank-bm25`, `nltk`
- `langchain-ollama`
- `langchain-text-splitters`

## Solução de problemas

Erro de conflito de dependências (`ResolutionImpossible`):
- recrie o ambiente virtual;
- atualize `pip/setuptools` (evite atualizar `wheel` sem necessidade);
- reinstale com `pip install -r requirements.txt`.

Erro ao usar Google:
- instale `langchain-google-genai`;
- configure `GOOGLE_API_KEY`;
- defina `LLM_PROVIDER=google`.
