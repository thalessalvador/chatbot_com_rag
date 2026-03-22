# Projeto Final PLN - Chatbot Tributario com RAG Hibrido

Projeto de consulta a pareceres tributarios com pipeline completo de:
- scraping de documentos;
- transformacao para Markdown;
- ingestao semantica juridica (chunking por secao);
- indexacao hibrida (Dense + BM25 + RRF);
- interface de chat com Streamlit.

## Requisitos
- Python 3.11
- Ollama instalado (para provider local)
- LibreOffice instalado (para converter formatos legados `.doc/.xls/.ppt`)

## Estrutura
```text
app/app.py
src/pipeline.py
src/rag_core.py
src/app_config.py
src/logging_config.py
config.yaml
.env
requirements.txt
data/raw/
data/transformation/
data/processed/
data/index/
logs/
```

## Configuracao

### 1) `config.yaml` (fonte principal)
Todos os parametros operacionais ficam em `config.yaml`, incluindo:
- `llm` (provider, modelo, URL do Ollama, parametros de geracao);
- `embeddings`;
- `evaluation`;
- `retrieval`;
- `scraping`;
- `transform`;
- `legal_chunking`;
- `logging`;
- `ui`.

### 2) `.env` (apenas segredo)
Atualmente o `.env` e usado somente para chave da API Google:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

### Modelo de embeddings
O modelo de embeddings ativo e controlado por `config.yaml -> embeddings.model`.

No estado atual do projeto, o modelo configurado e:
- `stjiris/bert-large-portuguese-cased-legal-tsdae`

Sempre que `embeddings.model` for alterado em `config.yaml`, e necessario reexecutar:

```powershell
python src/pipeline.py --step index
```

Isso recria o indice vetorial do ChromaDB com o novo espaco vetorial.

O arquivo do Golden Set e o valor padrao de `Recall@k` tambem ficam centralizados em:
- `evaluation.golden_file`
- `evaluation.recall_k`

A resposta padrao de ausencia de contexto fica em:
- `rag.no_context_response`

## Instalar e rodar (Windows / PowerShell)

```powershell
cd C:\Projetos\Pos_ia\PLN\chatbot_com_rag

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools
pip install -r requirements.txt
```

## Execucao completa do pipeline

```powershell
python src/pipeline.py --step scraping
python src/pipeline.py --step transform
python src/pipeline.py --step ingest
python src/pipeline.py --step index
python src/pipeline.py --step evaluate --golden-file Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx --k 5
```

Depois subir o chatbot:

```powershell
streamlit run app/app.py
```

URL local: `http://localhost:8501`

## O que cada etapa faz

### `--step scraping`
- Coleta links da pagina configurada em `scraping.base_url`.
- Baixa arquivos para `data/raw/`.
- Gera manifesto: `data/raw/scraping_manifest.jsonl`.

### `--step transform`
- Converte arquivos para Markdown em `data/transformation/`.
- Gera manifesto: `data/transformation/transform_manifest.jsonl`.
- Converte formatos legados `.doc/.xls/.ppt` via LibreOffice (`soffice`).
- Normaliza headings juridicos (opcional por config).

### `--step ingest`
- Le os `.md` transformados.
- Segmenta por secoes juridicas (relatorio, fundamentacao, conclusao etc.).
- Quebra secoes longas por tokens (`legal_chunking.max_tokens`).
- Enriquece metadados (ex.: `assunto`, `secao`, `normas_citadas`, `tributos_citados`).
- Salva chunks em `data/processed/chunks.json`.

### `--step index`
- Gera embeddings Dense (SentenceTransformers).
- Grava indice vetorial no ChromaDB (`data/index/chroma_db`).
- Gera indice BM25 (`data/index/bm25_index.pkl`).
- Deve ser reexecutado sempre que `embeddings.model` for alterado.

### `--step evaluate`
- Executa a avaliacao do retriever com a metrica `Recall@k`.
- Le o Golden Set em Excel informado por `--golden-file`.
- Testa os modos `dense`, `sparse` e `hybrid`.
- Exibe e registra no log o resultado final por modo.
- Usa os valores padrao definidos em `config.yaml -> evaluation`.

Exemplo:

```powershell
python src/pipeline.py --step evaluate --golden-file Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx --k 5
```

Pre-requisitos:
- indice vetorial ja gerado em `data/index/chroma_db`;
- indice BM25 ja gerado em `data/index/bm25_index.pkl`;
- arquivo do Golden Set disponivel na raiz do projeto ou informado via `--golden-file`.

## LLMs suportados

Definido em `config.yaml -> llm.provider`:
- `ollama`
- `google`

No Streamlit, o usuario tambem pode escolher provider/modelo na sidebar.

## GPU x CPU

- Embeddings usam `embeddings.device` (`cpu` ou `cuda`).
- Ollama usa parametros `llm.ollama_*` (incluindo `ollama_num_gpu` e `ollama_num_thread`).
- BM25/tokenizacao e partes de preprocessing rodam em CPU (esperado).

## Logging

Logging centralizado em console + arquivo rotativo.

Configurado em `config.yaml -> logging`:
- `level`
- `file`
- `max_bytes`
- `backup_count`

Log padrao: `logs/app.log`

## LibreOffice (transformacao de formatos legados)

Se `transform` falhar com `.doc/.xls/.ppt`, valide:

```powershell
Get-Command soffice
```

Se nao localizar, configure caminho explicito em `config.yaml`:

```yaml
transform:
  libreoffice_cmd: "C:\\Program Files\\LibreOffice\\program\\soffice.com"
```

## Troubleshooting rapido

### 1) Erro de metadata no Chroma (`Expected metadata value ... got []`)
Ja tratado no codigo com sanitizacao de metadados no `index`.

### 2) Embeddings sem GPU
Verifique:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

### 3) Resposta cortada no chat
Aumente no `config.yaml`:
- `llm.ollama_num_predict`
- e, se necessario, `llm.ollama_num_ctx`

### 4) Telemetria do Chroma aparecendo em log
Nao bloqueia funcionamento. O projeto ja tenta desativar telemetria anonima no client e por ambiente.

## Observacoes de versionamento

- `data/` esta no `.gitignore` (artefatos locais de pipeline).
- `logs/` tambem esta ignorado.
