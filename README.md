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

### `--step evaluate`
- Etapa ainda placeholder no estado atual.

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
