# Projeto Final PLN - Chatbot Tributario com RAG Hibrido

Projeto de consulta a pareceres tributarios com pipeline completo de:
- scraping de documentos;
- transformacao para Markdown;
- ingestao semantica juridica (chunking por secao);
- indexacao hibrida (Dense + BM25 + RRF);
- interface de chat com Streamlit.

Relatorio de validacao, metricas Recall@k, experimentos de embedding e rubrica qualitativa (PR5–PR7): **[RELATORIO_PR7.md](RELATORIO_PR7.md)**.

## Requisitos
- Python 3.11
- Ollama instalado (para provider local)
- LibreOffice instalado (para converter formatos legados `.doc/.xls/.ppt`)

## Estrutura
```text
RELATORIO_PR7.md
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

### Reranking opcional
O reranking fica em `config.yaml -> retrieval.reranking`.
Ele permanece desligado por padrao (`enabled: false`) e deve ser ativado apenas para experimento ou comparacao de metricas.

Com `enabled: true`, o fluxo de recuperacao no modo `hybrid` passa a ser:
- busca dense;
- busca sparse;
- fusao RRF;
- reranking dos melhores candidatos;
- corte final em `top_k`.

Parametros:
- `enabled`: ativa ou desativa o reranking;
- `model`: modelo Hugging Face usado no reranker;
- `device`: `cpu` ou `cuda`;
- `candidate_pool_size`: quantidade de candidatos vindos da fusao RRF que serao reordenados.

Exemplo:

```yaml
retrieval:
  top_k: 5
  rrf_k: 60
  reranking:
    enabled: false
    model: BAAI/bge-reranker-v2-m3
    device: cuda
    candidate_pool_size: 20
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
- Se `retrieval.reranking.enabled=true`, o reranking so afeta o modo `hybrid`.
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

### Streamlit com Ollama: VRAM e concorrencia

Ao rodar `streamlit run app/app.py` com **`llm.provider: ollama`**, o **SentenceTransformer** (embeddings na consulta) e o **servidor Ollama** podem disputar a **mesma GPU** e a **VRAM** ao mesmo tempo. Isso pode deixar a geracao muito lenta ou parecer “travada”, sobretudo em placas com pouca memoria (ex.: 8 GB).

**Recomendacao pratica (escolha uma abordagem ou combine com testes):**

1. **Embeddings em CPU durante o uso do chat (mais simples)**  
   Antes de subir o Streamlit, em `config.yaml`, defina:
   ```yaml
   embeddings:
     device: cpu
   ```
   A indexacao (`--step index`) pode continuar a usar `cuda` se desejares (alteras para `cpu` so na hora de testar o chat, ou mantens `cpu` sempre que priorizares o Ollama na GPU). O encode de **uma pergunta** por vez na CPU costuma ser aceitavel e **liberta VRAM** para o modelo do Ollama.  
   **Importante:** apos mudar `embeddings.model` ou `device` para alinhar com o indice, reinicie o Streamlit e use *Clear cache* se necessario (ver secao de cache na documentacao do projeto).

2. **Ajustar `llm.ollama_num_gpu` para melhor reparticao**  
   Em `config.yaml -> llm`, o parametro `ollama_num_gpu` e repassado ao Ollama (camadas/GPUs usadas no backend). Em **maquinas com varias GPUs**, ajuste para a quantidade adequada ao teu hardware e a forma como queres isolar Ollama vs. PyTorch (tambem podes usar `CUDA_VISIBLE_DEVICES` no ambiente do Ollama ou do Python, conforme o teu cenario).  
   Em **uma unica GPU**, valores muito altos (ex.: `9999`) pedem o maximo de offload para a GPU; se a VRAM enche com o modelo de embeddings ainda em `cuda`, reduzir `ollama_num_gpu` pode fazer parte das camadas do LLM correr em CPU, **libertando VRAM** — ao custo de velocidade no Ollama. Experimenta valores intermediarios conforme o modelo e a placa.

**Resumo:** com Ollama local, a opcao mais estavel em GPUs com pouca VRAM e, em geral, **`embeddings.device: cpu`** no momento de usar o Streamlit; alternativamente (ou em paralelo), **afina `ollama_num_gpu`** e o ambiente CUDA para equilibrar os dois cargos.

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
