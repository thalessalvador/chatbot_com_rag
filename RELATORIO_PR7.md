# Relatorio de Validação e Avaliacao do RAG (PR5, PR6 e PR7)

## 1. Visão Geral

Este documento apresenta os resultados da avaliação formal do sistema de Retrieval-Augmented Generation (RAG) para Pareceres Tributários da SEFAZ-GO. O objetivo deste projeto foi medir matematicamente a qualidade da recuperação de informação (PR5), consolidar a exposição dos modos de busca na interface (PR6) e registrar os artefatos finais da etapa experimental (PR7).

## 2. Desenvolvimento e Organização

Foram utilizados dados de pareceres públicos disponíveis no [endpoint principal](https://appasp.economia.go.gov.br/pareceres/) para desenvolvimento deste trabalho.

A partir desses documentos foram estabelecidas e desenvolvidas estratégias de transformação de dados, chunking e indexação a fim de geração da RAG.

Foram utilizados os modelos **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) básico e o **Bert Jurídico** `stjiris/bert-large-portuguese-cased-legal-tsdae` - encoder especialista do STJ (Superior Tribunal de Justiça) - como bases para o treinamento com os documentos.

### 2.1 Processo de chunking (ingestão jurídica)

Esta seção documenta **o que é necessário** para reproduzir e justificar a segmentação dos pareceres antes da indexação. A implementação está desenvolvida em `src/pipeline.py` (função `run_ingest` e auxiliares), com parâmetros em `config.yaml` → `legal_chunking` e `transform`.

#### 2.1.1 Entrada e pré-requisitos

1. **Markdown por documento** em `data/transformation/`, listado na transformação (etapa `--step transform`). Cada arquivo pode incluir **front matter YAML** inicial (`---` … `---`) com metadados (`doc_id`, `titulo`, `fonte`, `data`, `tipo`, `assunto`). Garantimos dessa forma a estrutura e melhor posicionamento dos chunks.
2. **Normalização de estrutura**: com `transform.normalize_legal_headings: true`, os títulos frequentes de parecer (EMENTA, RELATÓRIO, FUNDAMENTAÇÃO, CONCLUSÃO, DISPOSITIVO, numeração romana, etc.) são padronizados como headings `##` antes da ingestão, o que **melhora a detecção de seções do documento** descrita abaixo.
3. **Golden Set**: a partir dos documentos obtidos foi possível a escrita de perguntas que atestam a eficiência do RAG. No contexto dessas perguntas existem 3 tipos: as que aparecem no contexto dos documentos (dentro do corpus), as que usam multi-trechos (perguntas que envolvem possivelmente mais de um chunk) e as que não estão presentes no contexto (fora do corpus) - todas possuem como objetivo garantir o Recall@, detectar aluninações, avaliar a capacidade da RAG em escolher os trechos mais correspondentes e da LLM em escrever a resposta. No total 29 perguntas foram desenvolvidas.

#### 2.1.2 Primeiro nível: seções jurídicas

O corpo do Markdown é percorrido **linha a linha** (`_detect_legal_sections`):

- **Padrões de cabeçalho** alinhados a documentos do Estado de Goiás: EMENTA, RELATÓRIO, FUNDAMENTAÇÃO, CONCLUSÃO, DISPOSITIVO (incluindo variantes com/sem acentos e com numeração tipo `I - RELATÓRIO`).
- **Qualquer linha que comece por `#`** inicia nova seção; o texto acumulado até ao próximo limiar é guardado como um bloco.
- Cada bloco recebe um rótulo (`ementa`, `relatorio`, `fundamentacao`, `conclusao`, `dispositivo` ou `secao_geral`) para fins de **filtro semântico e de metadados** no RAG.

**Justificativa:** pareceres tributários seguem uma **ordem argumentativa estável**; segmentar por essas seções evita misturar, num único chunk longo, relatório descritivo do problema que o parecer busca responder, fundamentação jurídica e conclusão, o que prejudicaria tanto a recuperação por embedding como a formulação da resposta pela LLM.

#### 2.1.3 Segundo nível: sub-chunks por tamanho (tokens)

Seções longas são ainda divididas (`_split_section_into_subchunks`):

- **Contagem de tokens:** usa `nltk.word_tokenize` quando disponível; caso contrário, **fallback** por tokens alfanuméricos (`\w+`).
- **`legal_chunking.max_tokens`** (no projeto: **700**): tamanho máximo da janela de tokens.
- **`legal_chunking.overlap_tokens`** (no projeto: **140**): sobreposição entre janelas consecutivas; o passo é executado por `max_tokens - overlap_tokens` (mínimo 1), gerando **janela deslizante** entre chunks vizinhos.

**Nota técnica:** a divisão é **por token**, não por parágrafos nem por `RecursiveCharacterTextSplitter`; prioriza **limite de contexto** e reprodutibilidade em texto jurídico denso. Listas e tabelas herdadas do Markdown ficam no fluxo linear da seção — podem ser cortadas a meio da janela.

#### 2.1.4 Indexação, metadados e texto enriquecido

- **`chunk_id`:** `{doc_id}#chunk_` + índice sequencial de quatro dígitos por documento (`0000`, `0001`, …).
- **Textos extras extraídos do sub-chunk:** `normas_citadas` (regex sobre artigos, leis, CTN, CF/88, etc.) e `tributos_citados` (keywords: ICMS, IPVA, ITCD, ISS, …).
- **Texto indexado no denso (`texto`):** versão **enriquecida** (`_build_enriched_text`) com prefixo explícito: tipo de documento, `doc_id`, título, assunto, seção, normas e tributos agregados, seguido do texto. Isto **reforça o sinal semântico** do embedding com o contexto normativo. O **BM25** e o contexto apresentado ao LLM podem usar `texto_bruto` / metadados conforme o fluxo em `rag_core.py`.

#### 2.1.5 Saída e comando

- **Arquivos gerados:** `data/processed/chunks.json` (lista de objetos com `chunk_id`, `doc_id`, `texto`, `texto_bruto`, `metadados`).
- **Comando:** `python src/pipeline.py --step ingest` (após `transform`).

## 3. Trilha A: Recuperação híbrida

Esta seção descreve a execução da Trilha A: motores **denso** e **esparso** híbridos, **fusão RRF**, parâmetros de configuração e **interpretação quantitativa** (Recall@k). A implementação está disponível em `src/rag_core.py` (`HybridRAG.retrieve`) e os hiperparâmetros em `config.yaml` → `retrieval`.

Foram comparadas três estratégias de recuperação:

- Denso (embeddings): recuperação semantica por vetores.
- Esparso (BM25): recuperação lexical por frequência exata de termos.
- Hibrido (RRF): combinação entre denso e esparso por Reciprocal Rank Fusion (RRF). - Trilha A abaixo.

### 3.1 Recuperação densa (embeddings)

- O texto de cada chunk (versão enriquecida para indexação) foi previamente vetorizado com **`SentenceTransformer`** na etapa `--step index`, armazenado no **ChromaDB** (`data/index/chroma_db`).
- Em cada pergunta, o mesmo modelo codifica a **query** num vector de consulta; o Chroma devolve os `chunk_id` mais próximos por similaridade.
- O **espaço semântico** depende exclusivamente de `embeddings.model`. Nesse trabalho é feita a gravação do processo de indexação e estatísticas dos modelos em `recall_por_k.json` (Bert e MiniLM). Trocar o modelo exige **reindexar** (pipeline com `--step index`).

### 3.2 Recuperação esparsa (BM25)

- Utiliza **BM25Okapi** (`rank_bm25`) sobre **todos** os chunks carregados a partir de `data/index/bm25_index.pkl` (gerado no `--step index` com o mesmo `chunks.json` que o denso).
- A query é **tokenizada** (`nltk.word_tokenize` com fallback por `\w+`); a pontuação BM25 gera um ranking por **coincidência lexical** (termos raros e repetidos no documento possuem pesos distintos e não busca similaridade entre palavras).
- **Não** usa embeddings: por isso, nos experimentos deste relatório, as linhas **esparso** do JSON são **idênticas** para MiniLM e BERT.

### 3.3 Fusão híbrida (RRF) e “deduplicação”

No modo `hybrid` aplica **Reciprocal Rank Fusion (RRF)** sobre **duas listas ordenadas de `chunk_id`**:

1. Lista **A** (denso): ordenação por proximidade no Chroma.  
2. Lista **B** (esparso): ordenação por score BM25 decrescente.

Para cada lista, cada posição `rank` (base 0) contribui para o identificador `doc_id` com o termo **1 / (k_rrf + rank + 1)**. Se o mesmo `chunk_id` aparece nas **duas** listas, os dois termos **somam** num score RRF agregado — efeito natural de **unificar** o mesmo parecer sem duplicar linhas no ranking final.

Hiperparâmetro **`retrieval.rrf_k`** (no projeto: **60**, configurável por ambiente como `RRF_K`): quanto **maior** `k_rrf`, mais **suave** é a pena por rank (contribuições dos primeiros lugares diferenciam-se menos dos seguintes).

**Dimensão do conjunto candidato** (sem reranker): é escolhido um `max(top_k * 2, …)` resultados por pergunta antes da fusão; depois ordena-se por score RRF e corta-se em **`retrieval.top_k`** (predefinição **5** na configuração do repositório, alinhada à avaliação Recall@5). Com **`retrieval.reranking.enabled: false`** (estado actual), **não** há passo extra de cross-encoder após o RRF.

### 3.4 Quando o híbrido ganha ou perde (números MiniLM, golden com 29 casos)

Valores de `docs/dados/recall_por_k.json` (primeira entrada de `runs` — MiniLM):

| k | Denso | Esparso | Híbrido | Nota |
|---|-------|---------|---------|------|
| 3 | 24,14% (7/29) | 65,52% (19/29) | 55,17% (16/29) | Híbrido **supera** o denso (+31,03 p.p.); fica **abaixo** do só BM25 (−10,35 p.p.). |
| 5 | 27,59% (8/29) | 72,41% (21/29) | 68,97% (20/29) | Híbrido **supera** o denso (+41,38 p.p.); fica **ligeiramente abaixo** do esparso (−3,44 p.p.). |
| 10 | 44,83% (13/29) | 75,86% (22/29) | 82,76% (24/29) | Híbrido **supera** ambos; o denso **melhora** face a k=5 neste conjunto. |

**Interpretação:** o **denso MiniLM** é o elo mais fraco neste golden enquanto o **BM25** já traz forte cobertura lexical. O RRF **corrige** parte da lacuna do denso em k=5 e k=10 ao promover documentos que aparecem bem posicionados em **qualquer** uma das listas. 

Com o **BERT legal** no denso (segunda entrada de `runs`), o denso permanece mais fraco que o MiniLM em k=3 e k=5; o híbrido **aproveita** o esparso — em Recall@5 o BERT legal (**72,41%**, 21/29) **supera** o MiniLM (**68,97%**, 20/29); em Recall@3 o híbrido BERT (**58,62%**, 17/29) fica **abaixo** do esparso (**65,52%**, 19/29).

### 3.5 Interface

Optamos pelo desenvolvimento com Streamlit em interface web: nessa visualização é possivel escolher entre os modos **Híbrido**, **Denso** e **Esparso** e comparar os trechos recuperados para a mesma pergunta, em linha com a Trilha A - demonstrando não apenas a respostas mas o Recall@5 dos melhores trechos.
Não há persistência de memória, portanto novas perguntas podem ser feitas em sequência umas às outras.

## 4. 1º Teste: Recall 3, 5 e 10 - Motor MiniLM

Para avaliação do modelo foi construído um script capaz de rodar os modelos em k-posições necessárias. Para tal, rode (para k IN (3, 5 e 10)). O script obedece o modelo citado em `embeddings.model` e depende de `--step index`:

```powershell
python scripts/run_recall_at_k_values.py --k-values 3,5,10
```

**MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`): **Recall@k** por modo de recuperação. Valores extraídos de `docs/dados/recall_por_k.json` (primeira entrada de `runs`; `updated_at`: `2026-04-02T16:06:29.615213+00:00`), com **29** perguntas válidas no denominador.

| k | Modo | Recall (%) | acertos/total |
|---|------|------------|---------------|
| 3 | Denso | 24,14 | 7/29 |
| 3 | Esparso | 65,52 | 19/29 |
| 3 | Hibrido | 55,17 | 16/29 |
| 5 | Denso | 27,59 | 8/29 |
| 5 | Esparso | 72,41 | 21/29 |
| 5 | Hibrido | 68,97 | 20/29 |
| 10 | Denso | 44,83 | 13/29 |
| 10 | Esparso | 75,86 | 22/29 |
| 10 | Hibrido | 82,76 | 24/29 |

Resumo **Recall@5** (baseline MiniLM, conforme JSON):

- Modo Denso: **27,59%** (8/29)
- Modo Esparso: **72,41%** (21/29)
- Modo Hibrido: **68,97%** (20/29)

## 5. 2º Teste: Recall 3, 5 e 10 - Motor Bert Jurídico

Comparação **Recall@k**: colunas **MiniLM** e **BERT legal** correspondem, respetivamente, à primeira e segunda entradas de `runs` em `docs/dados/recall_por_k.json` (mesmo `updated_at` acima), com **29** casos válidos.

| k | Modo | MiniLM — Recall (%) | MiniLM — acertos/total | BERT legal — Recall (%) | BERT legal — acertos/total |
|---|------|---------------------|------------------------|-------------------------|----------------------------|
| 3 | Denso | 24,14 | 7/29 | 10,34 | 3/29 |
| 3 | Esparso | 65,52 | 19/29 | 65,52 | 19/29 |
| 3 | Hibrido | 55,17 | 16/29 | 58,62 | 17/29 |
| 5 | Denso | 27,59 | 8/29 | 10,34 | 3/29 |
| 5 | Esparso | 72,41 | 21/29 | 72,41 | 21/29 |
| 5 | Hibrido | 68,97 | 20/29 | 72,41 | 21/29 |
| 10 | Denso | 44,83 | 13/29 | 20,69 | 6/29 |
| 10 | Esparso | 75,86 | 22/29 | 75,86 | 22/29 |
| 10 | Hibrido | 82,76 | 24/29 | 82,76 | 24/29 |

Experimento: alteração controlada do modelo de embeddings para `stjiris/bert-large-portuguese-cased-legal-tsdae`.

**Recall@5 (BERT legal), conforme JSON:**

- Denso: **10,34%** (3/29)
- Esparso: **72,41%** (21/29)
- Hibrido: **72,41%** (21/29)

## 6. Conclusao Técnica

![Recall@3 — MiniLM vs BERT legal](docs/imagens/recall_k3.png)

![Recall@5 — MiniLM vs BERT legal](docs/imagens/recall_k5.png)

![Recall@10 — MiniLM vs BERT legal](docs/imagens/recall_k10.png)

Com o golden set atual (**29** casos válidos neste JSON), o embedding jurídico permanece **abaixo** do MiniLM no modo **denso** em todos os **k**: Recall@3 **10,34%** (3/29) vs **24,14%** (7/29); Recall@5 **10,34%** (3/29) vs **27,59%** (8/29); Recall@10 **20,69%** (6/29) vs **44,83%** (13/29). No modo **esparso**, os dois modelos **empatam** em cada **k** (idem acertos/total: 19/29, 21/29 e 22/29, por o BM25 não depender do modelo de embeddings). No modo **híbrido**, Recall@5 regista **68,97%** (20/29) no MiniLM e **72,41%** (21/29) no BERT legal; em Recall@10 **ambos** atingem **82,76%** (24/29).

Em síntese, com estes números consolidados em `recall_por_k.json`, o BERT legal **supera** ligeiramente o MiniLM em Recall@5 no híbrido (**72,41%** vs **68,97%**). A **Trilha A** mantém-se justificada porque o **híbrido** supera, em geral, denso e esparso isolados; a escolha do embedding deve equilibrar **custo/tempo** (BERT maior).

## 7. Analise por rubrica qualitativa

Esta seção incorpora as anotações do arquivo **`rubrica_qualitativa.xlsx`** (na raiz do projeto; folha `rubrica_qualitativa`), com **15** linhas de avaliação: **5** perguntas do golden set (linhas 1, 2, 3, 23 e 32) cruzadas com os **três** modos de recuperação (**Híbrido**, **Denso**, **Esparso** / BM25). Foi utilizado o modelo `stjiris/bert-large-portuguese-cased-legal-tsdae` para o teste de rubricas.

O trabalho de rubrica foi feito de forma manual, interagindo diretamente com o modelo via navegador no Streamlit. Os textos foram avaliados em totalidade buscando analisar os critérios da próxima seção.

### 7.1 Critérios utilizados

Para cada execução foram registadas notas (escala numérica em geral de 1 a 5, salvo `N/D` quando o critério não se aplicava) e comentários livres:

| Critério | Significado na rubrica|
|----------|-----------------------------------------------|
| **Groundedness** | A resposta permanece ancorada nos trechos recuperados? |
| **Correção** | Alinhamento com o conteúdo normativo/factual esperado para a pergunta. |
| **Citações** | Uso adequado das referências `[[TRECHO_n]]` / coerência com as fontes. |
| **Alucinação** | Nota alta = pouca ou nenhuma invenção em relação ao corpus |
| **Recusa** | Quando aplicável (ex.: tema fora do corpus), capacidade de recusar em vez de inventar. |

As **observações** textuais da folha resumem-se abaixo, preservando o critério do avaliador.

### 7.2 Síntese por pergunta e modo

**Pergunta 1 (linha golden 1)** — *O que é fundeinfra?*

| Modo | G | C | Cit | Alu | Rec | Observações (resumo) |
|------|---|---|-----|-----|-----|----------------------|
| Híbrido | 5 | 4 | 5 | 5 | N/D | Cita o Fundeinfra mas erra número da lei, não cita ICMS; resposta longa e desfocada do cerne da pergunta. |
| Denso | 1 | 1 | 1 | 5 | N/D | Não encontrou resposta útil (falha de recuperação / resposta vazia). |
| **Esparso (BM25)** | **5** | **5** | **5** | **5** | N/D | **Melhor resposta global** na rubrica. |

**Pergunta 2 (linha golden 2)** — *Empresa de SP, duas vendas de máquinas (não contribuinte R$ 100 mil e contribuinte R$ 200 mil): quanto pagar de DIFAL para Goiás?*

| Modo | G | C | Cit | Alu | Rec | Observações (resumo) |
|------|---|---|-----|-----|-----|----------------------|
| Híbrido | 5 | 2 | 3 | 5 | N/D | Raciocínio invertido; acertou o parecer; errou o cálculo. |
| Denso | 5 | 3 | 3 | 5 | N/D | Acertou parecer e cálculo, mas não usou bem o parecer na formulação da resposta. |
| Esparso (BM25) | 5 | 2 | 3 | 5 | N/D | Usou o parecer de forma coerente; errou cálculo e base de cálculo. |

Neste caso a **correção** ficou ligeiramente melhor no **Denso**, mas todos falham em parte no cálculo ou na articulação; a rubrica destaca sobretudo **grounding** estável nos três.

**Pergunta 3 (linha golden 3)** — *A construtora é obrigada a fazer a Escrituração Fiscal Digital?*

| Modo | G | C | Cit | Alu | Rec | Observações (resumo) |
|------|---|---|-----|-----|-----|----------------------|
| Híbrido | 5 | 2 | 5 | 5 | N/D | Coerente com o documento recuperado, mas não coincide com a resposta esperada do golden. |
| Denso | 1 | 1 | 1 | 1 | N/D | Não encontrou resposta. |
| **Esparso (BM25)** | **5** | **2** | **5** | **5** | N/D | Mesma linha do híbrido quanto ao alinhamento com o gabarito, com resposta **mais sucinta e direta**. |

**Pergunta 23 (linha golden 23)** — *Caminhonete é veículo utilitário?*

| Modo | G | C | Cit | Alu | Rec | Observações (resumo) |
|------|---|---|-----|-----|-----|----------------------|
| **Híbrido** | **5** | **5** | **5** | **1** | N/D | **Melhor conjunto** na rubrica: chunks fortes no top-3, resposta coesa (IPVA, órgão competente). |
| Denso | 2 | 1 | 5 | 1 | N/D | Parecer correto fora do top-5; trechos sem correlação temática (ex.: combustíveis). |
| Esparso (BM25) | 5 | 4 | 5 | 5 | N/D | Top-4 alinhados ao parecer esperado; resposta um pouco aberta quanto ao “modelo”; falta explicitar competência estadual. |

**Pergunta 32 (linha golden 32)** — *Quais requisitos da ANVISA para registro de cosmético importado (petição, taxas e documentação técnica)?* (intencionalmente **fora do corpus** tributário GO)

| Modo | G | C | Cit | Alu | Rec | Observações (resumo) |
|------|---|---|-----|-----|-----|----------------------|
| Híbrido | 5 | 1 | 5 | 1 | 1 | Alucinação: afirma sem corpus ANVISA; texto ainda “colado” a chunks errados. |
| **Denso** | **5** | **5** | **5** | **5** | **5** | **Recusa clara** pela ausência de informação no corpus (melhor comportamento de segurança). |
| Esparso (BM25) | 1 | 1 | 1 | 1 | 1 | Alucinação grave (incl. troca de idioma na anotação). |

*(Legenda: G = groundedness, C = correcao, Cit = citacoes, Alu = alucinacao, Rec = recusa.)*

### 7.3 Justificativa da melhor escolha do modelo **BM25** (modo esparso)

Com base **somente** nas anotações de `rubrica_qualitativa.xlsx` (raiz do projeto) e em alinhamento com o **Recall@k** já reportado (onde o **esparso** supera o **denso** em vários k):

1. **Termos jurídicos e siglas** — Perguntas como *Fundeinfra* ou *Escrituração Fiscal Digital* beneficiam da **coincidência lexical** com os pareceres. O **BM25** promove documentos em que esses tokens aparecem com frequência e relevância estatística; na rubrica, o **denso falhou por completo** nas perguntas 1 e 3 (notas mínimas e “não encontrou resposta”), enquanto o **esparso** manteve **groundedness e citações altas** e foi classificado como **melhor resposta** no caso Fundeinfra.

2. **Robustez quando o embedding denso desvia** — No caso **caminhonete / veículo utilitário**, o avaliador registra que o **denso** recuperou trechos **fora do tema** (ainda no “universo” tributário, mas irrelevantes). O **BM25** manteve **quatro** chunks do parecer esperado no topo, com **menos dispersão semântica** para esta formulação da pergunta. Isso acontece pois o modo denso buscava sinônimos ou termos semelhantes estatisticamente, enquanto o esparso busca por relevância. 

3. **Consistência com a métrica automática** — O **Recall@5** esparso (**72,41%**, 21/29) já indicava vantagem sobre o denso (**27,59%**, 8/29) no mesmo golden; a rubrica **confirma qualitativamente** que, em perguntas **ancoradas no léxico** dos pareceres de Goiás, o **esparso é frequentemente o modo mais seguro** quando o utilizador não reformula a pergunta em linguagem “próxima” do espaço semântico do modelo de embeddings.

4. **Ressalva obrigatória: temas fora do corpus** — Inicialmente rodamos os testes com ambos os modelos apenas com **engenharia de prompt** com **few-shot prompting**. Essa abordagem não estava tratando totalmente os temas fora do corpus, gerando alucinação e respostas corretas mas fora do contexto restrito dos documentos. Dessa forma fizemos um **loop** que verifica a resposta da gerada pela LLM e reavalia, tendo em vista a pergunta inicial (*LLM as judge*)

Em síntese, para o **domínio deste chatbot** (pareceres tributários GO com perguntas ricas em **nomes de institutos, normas e fatos concretos**), a rubrica qualitativa **reforça o BM25** como **baseline forte** frente ao denso isolado, **coerente** com o Recall quantitativo, com a **exceção documentada** de perguntas **sem suporte no corpus**, onde o **denso** foi superior na **recusa**.


