# Relatorio de Validacao e Avaliacao do RAG (PR5, PR6 e PR7)

## 1. Visao Geral

Este documento apresenta os resultados da avaliacao formal do sistema de Retrieval-Augmented Generation (RAG) para Pareceres Tributarios da SEFAZ-GO. O objetivo desta fase foi medir matematicamente a qualidade da recuperacao de informacao (PR5), consolidar a exposicao dos modos de busca na interface (PR6) e registrar os artefatos finais da etapa experimental (PR7).

## 2. Artefatos Produzidos

Foi utilizado um Golden Set em Excel com perguntas de negocio e documento esperado, consumido diretamente pela etapa:

```powershell
python src/pipeline.py --step evaluate --golden-file Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx --k 5
```

Na execucao consolidada nesta fase, a avaliacao considerou **29** perguntas com gabarito utilizavel (linhas com `documento_esperado` preenchido; entradas `Nenhum` ou vazias ficam de fora do denominador do Recall). O ficheiro inclui tambem itens fora do corpus para rubrica de recusa, que nao entram nesta metrica.

## 3. Estrategias Avaliadas

Foram comparadas tres estrategias de recuperacao:

- Denso (embeddings): recuperacao semantica por vetores.
- Esparso (BM25): recuperacao lexical por frequencia exata de termos.
- Hibrido (RRF): combinacao entre denso e esparso por Reciprocal Rank Fusion.

## 4. Trilha A: arquitetura de recuperação híbrida

Esta secção cumpre o requisito de **documentar** a Trilha A: motores **denso** e **esparso**, **fusão RRF**, parâmetros de configuração e **interpretação quantitativa** (Recall@k) face ao denso isolado. A implementação de referência está em `src/rag_core.py` (`HybridRAG.retrieve`) e os hiperparâmetros em `config.yaml` → `retrieval`.

### 4.1 Recuperação densa (embeddings)

- O texto de cada chunk (versão enriquecida para indexação) foi previamente vectorizado com **`SentenceTransformer`** na etapa `--step index`, armazenado no **ChromaDB** (`data/index/chroma_db`).
- Em cada pergunta, o mesmo modelo codifica a **query** num vector de consulta; o Chroma devolve os `chunk_id` mais próximos por similaridade de coseno (ordem = rank denso).
- O **espaço semântico** depende exclusivamente de `embeddings.model` (neste relatório: comparação **MiniLM** vs **BERT legal** nos ficheiros `recall_por_k.json`). Trocar o modelo exige **reindexar** o denso.

### 4.2 Recuperação esparsa (BM25)

- Utiliza **BM25Okapi** (`rank_bm25`) sobre **todos** os chunks carregados a partir de `data/index/bm25_index.pkl` (gerado no `--step index` com o mesmo `chunks.json` que o denso).
- A query é **tokenizada** (`nltk.word_tokenize` com fallback por `\w+`); a pontuação BM25 gera um ranking por **coincidência lexical** (termos raros e repetidos no documento pesam mais, no espírito clássico de Okapi BM25).
- **Não** usa embeddings: por isso, nos experimentos deste relatório, as linhas **esparso** do JSON são **idênticas** para MiniLM e BERT legal (mesmos acertos/total por **k**).

### 4.3 Fusão híbrida (RRF) e “deduplicação”

No modo `hybrid`, o sistema **não** escolhe manualmente scores incomensuráveis entre denso e BM25. Aplica **Reciprocal Rank Fusion (RRF)** sobre **duas listas ordenadas de `chunk_id`**:

1. Lista **A** (denso): ordenação por proximidade no Chroma.  
2. Lista **B** (esparso): ordenação por score BM25 decrescente.

Para cada lista, cada posição `rank` (base 0) contribui para o identificador `doc_id` com o termo **1 / (k_rrf + rank + 1)**. Se o mesmo `chunk_id` aparece nas **duas** listas, os dois termos **somam** num score RRF agregado — efeito natural de **unificar** o mesmo parecer sem duplicar linhas no ranking final.

Hiperparâmetro **`retrieval.rrf_k`** (no projeto: **60**, configurável por ambiente como `RRF_K`): quanto **maior** `k_rrf`, mais **suave** é a pena por rank (contribuições dos primeiros lugares diferenciam-se menos dos seguintes). O código replica o padrão usual de fusão por ranks descrito na literatura de RAG híbrido.

**Dimensão do conjunto candidato** (sem reranker): recolhe-se um pool de até `max(top_k * 2, …)` resultados por ramo antes da fusão; depois ordena-se por score RRF e corta-se em **`retrieval.top_k`** (predefinição **5** na configuração do repositório, alinhada à avaliação Recall@5). Com **`retrieval.reranking.enabled: false`** (estado actual), **não** há passo extra de cross-encoder após o RRF.

### 4.4 Quando o híbrido ganha ou perde (números MiniLM, golden com 29 casos)

Valores de `docs/dados/recall_por_k.json` (primeira entrada de `runs` — MiniLM):

| k | Denso | Esparso | Híbrido | Nota |
|---|-------|---------|---------|------|
| 3 | 31,03% (9/29) | 65,52% (19/29) | 62,07% (18/29) | Híbrido **supera** o denso (+31 pontos percentuais); fica **ligeiramente abaixo** do só BM25 (−3,45 p.p.). |
| 5 | 41,38% (12/29) | 68,97% (20/29) | 72,41% (21/29) | Híbrido **supera** denso e **supera** esparso. |
| 10 | 41,38% (12/29) | 68,97% (20/29) | 82,76% (24/29) | Híbrido **supera** ambos; denso **estagna** face a k=5 neste conjunto. |

**Interpretação:** o **denso MiniLM** é o elo mais fraco neste golden; o **BM25** já traz forte cobertura lexical. O RRF **corrige** parte da lacuna do denso em k=5 e k=10 ao promover identificadores que aparecem bem posicionados em **qualquer** uma das listas. Em **k=3**, poucos postos no top final penalizam ligeiramente a fusão relativamente ao **só esparso** (o recall do híbrido fica entre denso e esparso).

Com o **BERT legal** no denso (segunda entrada de `runs`), o denso piora fortemente em k=3 e k=5; o híbrido continua a **aproveitar** o ramo esparso e aproxima-se do desempenho do híbrido MiniLM em Recall@5 (72,41% — empate), enquanto em Recall@3 o híbrido BERT (68,97%) **já ultrapassa** o esparso (65,52%) porque o contributo denso, embora fraco, reordena a fusão de forma mais favorável nesse **k**.

### 4.5 Interface

A mesma arquitectura está exposta no Streamlit (**secção 8** deste relatório): o utilizador pode alternar **Híbrido**, **Denso** e **Esparso** e comparar os trechos recuperados para a mesma pergunta, em linha com a Trilha A.

## 5. Baseline Confirmada

Baseline **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`): **Recall@k** por modo de recuperação. Valores extraídos de `docs/dados/recall_por_k.json` (primeira entrada de `runs`; `updated_at`: `2026-03-29T03:12:32.329432+00:00`), com **29** perguntas válidas no denominador.

| k | Modo | Recall (%) | acertos/total |
|---|------|------------|---------------|
| 3 | Denso | 31,03 | 9/29 |
| 3 | Esparso | 65,52 | 19/29 |
| 3 | Hibrido | 62,07 | 18/29 |
| 5 | Denso | 41,38 | 12/29 |
| 5 | Esparso | 68,97 | 20/29 |
| 5 | Hibrido | 72,41 | 21/29 |
| 10 | Denso | 41,38 | 12/29 |
| 10 | Esparso | 68,97 | 20/29 |
| 10 | Hibrido | 82,76 | 24/29 |

Resumo **Recall@5** (baseline MiniLM, conforme JSON):

- Modo Denso: **41,38%** (12/29)
- Modo Esparso: **68,97%** (20/29)
- Modo Hibrido: **72,41%** (21/29)

## 6. Experimento com Embedding Juridico

Comparação **Recall@k**: colunas **MiniLM** e **BERT legal** correspondem, respetivamente, à primeira e segunda entradas de `runs` em `docs/dados/recall_por_k.json` (mesmo `updated_at` acima), com **29** casos válidos.

| k | Modo | MiniLM — Recall (%) | MiniLM — acertos/total | BERT legal — Recall (%) | BERT legal — acertos/total |
|---|------|---------------------|------------------------|-------------------------|----------------------------|
| 3 | Denso | 31,03 | 9/29 | 17,24 | 5/29 |
| 3 | Esparso | 65,52 | 19/29 | 65,52 | 19/29 |
| 3 | Hibrido | 62,07 | 18/29 | 68,97 | 20/29 |
| 5 | Denso | 41,38 | 12/29 | 17,24 | 5/29 |
| 5 | Esparso | 68,97 | 20/29 | 68,97 | 20/29 |
| 5 | Hibrido | 72,41 | 21/29 | 72,41 | 21/29 |
| 10 | Denso | 41,38 | 12/29 | 31,03 | 9/29 |
| 10 | Esparso | 68,97 | 20/29 | 68,97 | 20/29 |
| 10 | Hibrido | 82,76 | 24/29 | 79,31 | 23/29 |

Experimento: alteração controlada do modelo de embeddings para `stjiris/bert-large-portuguese-cased-legal-tsdae`.

**Recall@5 (BERT legal), conforme JSON:**

- Denso: **17,24%** (5/29)
- Esparso: **68,97%** (20/29)
- Hibrido: **72,41%** (21/29)

## 7. Conclusao Tecnica

Para avaliacao do modelo foi construído um script capaz de rodar ambos os modelos em k-posições necessárias. Para tal, rode (para k IN (3, 5 e 10)):

```powershell
python scripts/run_recall_at_k_values.py --k-values 3,5,10
```

![Recall@3 — MiniLM vs BERT legal](docs/imagens/recall_k3.png)

![Recall@5 — MiniLM vs BERT legal](docs/imagens/recall_k5.png)

![Recall@10 — MiniLM vs BERT legal](docs/imagens/recall_k10.png)

Com o golden set atual (**29** casos válidos neste JSON), o embedding jurídico permanece **abaixo** do MiniLM no modo **denso** em todos os **k**: Recall@3 **17,24%** (5/29) vs **31,03%** (9/29); Recall@5 **17,24%** (5/29) vs **41,38%** (12/29); Recall@10 **31,03%** (9/29) vs **41,38%** (12/29). No modo **esparso**, os dois modelos **empatam** em cada **k** (idem acertos/total: 19/29, 20/29 e 20/29, por o BM25 não depender do modelo de embeddings). No modo **híbrido**, Recall@5 é **72,41%** (21/29) para ambos; em Recall@10 o **MiniLM** regista **82,76%** (24/29) e o **BERT legal** **79,31%** (23/29).

Em síntese, com estes números consolidados em `recall_por_k.json`, **não** há ganho do BERT sobre o MiniLM em Recall@5 no híbrido (empate). A **Trilha A** mantém-se justificada porque o **híbrido** supera, em geral, denso e esparso isolados; a escolha do embedding deve equilibrar **custo/tempo** (BERT maior) e a vantagem do **MiniLM** no **denso** e no **híbrido** a **k = 10** neste conjunto.

## 8. Implementacao na Interface

A interface em Streamlit foi desenhada para espelhar estas escolhas tecnicas no contacto com o utilizador. Na barra lateral, e possivel alternar entre **Hibrido**, **Denso** e **Esparso**, de modo que a mesma pergunta possa ser respondida com estrategias distintas sem sair da aplicacao. Isto aproxima a demonstracao oral do que o relatório quantifica: o utilizador ve a resposta com citacoes, pode abrir o painel de trechos recuperados e, ao mudar o modo, observa diretamente como mudam os chunks que sustentam o RAG.

## 9. Analise por rubrica qualitativa

Esta seção incorpora as anotações do ficheiro **`rubrica_qualitativa.xlsx`** (na raiz do projeto; folha `rubrica_qualitativa`), com **15** linhas de avaliação: **5** perguntas do golden set (linhas 1, 2, 3, 23 e 32) cruzadas com os **três** modos de recuperação (**Híbrido**, **Denso**, **Esparso** / BM25). Foi utilizado o modelo `stjiris/bert-large-portuguese-cased-legal-tsdae` para o teste de rubricas.

### 9.1 Critérios utilizados

Para cada execução foram registadas notas (escala numérica em geral de 1 a 5, salvo `N/D` quando o critério não se aplicava) e comentários livres:

| Critério | Significado na rubrica (uso neste trabalho) |
|----------|-----------------------------------------------|
| **Groundedness** | A resposta permanece ancorada nos trechos recuperados? |
| **Correção** | Alinhamento com o conteúdo normativo/factual esperado para a pergunta. |
| **Citações** | Uso adequado das referências `[[TRECHO_n]]` / coerência com as fontes. |
| **Alucinação** | Nota alta = pouca ou nenhuma invenção em relação ao corpus (conforme a convenção da folha). |
| **Recusa** | Quando aplicável (ex.: tema fora do corpus), capacidade de recusar em vez de inventar. |

As **observações** textuais da folha resumem-se abaixo, preservando o critério do avaliador.

### 9.2 Síntese por pergunta e modo

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

### 9.3 Justificativa da melhor escolha do modelo **BM25** (modo esparso)

Com base **somente** nas anotações de `rubrica_qualitativa.xlsx` (raiz do projeto) e em alinhamento com o **Recall@k** já reportado (onde o **esparso** supera o **denso** em vários k):

1. **Termos jurídicos e siglas** — Perguntas como *Fundeinfra* ou *Escrituração Fiscal Digital* beneficiam da **coincidência lexical** com os pareceres. O **BM25** promove documentos em que esses tokens aparecem com frequência e relevância estatística; na rubrica, o **denso falhou por completo** nas perguntas 1 e 3 (notas mínimas e “não encontrou resposta”), enquanto o **esparso** manteve **groundedness e citações altas** e foi classificado como **melhor resposta** no caso Fundeinfra.

2. **Robustez quando o embedding denso desvia** — No caso **caminhonete / veículo utilitário**, o avaliador registra que o **denso** recuperou trechos **fora do tema** (ainda no “universo” tributário, mas irrelevantes). O **BM25** manteve **quatro** chunks do parecer esperado no topo, com **menos dispersão semântica** para esta formulação da pergunta.

3. **Consistência com a métrica automática** — O **Recall@5** esparso (**68,97%**, 20/29) já indicava vantagem sobre o denso (**41,38%**, 12/29) no mesmo golden; a rubrica **confirma qualitativamente** que, em perguntas **ancoradas no léxico** dos pareceres de Goiás, o **esparso é frequentemente o modo mais seguro** quando o utilizador não reformula a pergunta em linguagem “próxima” do espaço semântico do modelo de embeddings.

4. **Ressalva obrigatória: temas fora do corpus** — Na pergunta **ANVISA**, o **BM25** obteve **piores** notas em todos os critérios e **alucinação** explícita na rubrica; o **denso** foi o único a **recusar** corretamente. Logo, a “melhor escolha” do BM25 **não é universal**: para **consultas claramente ausentes** do acervo, o relatório recomenda **priorizar o modo denso** (ou políticas de recusa no LLM) e tratar o **esparso** como **complementar** no **híbrido**, não como substituto cego em cenários de *out-of-domain*.

Em síntese, para o **domínio deste chatbot** (pareceres tributários GO com perguntas ricas em **nomes de institutos, normas e fatos concretos**), a rubrica qualitativa **reforça o BM25** como **baseline forte** frente ao denso isolado, **coerente** com o Recall quantitativo, com a **exceção documentada** de perguntas **sem suporte no corpus**, onde o **denso** foi superior na **recusa**.

## 10. Processo de chunking (ingestão jurídica)

Esta seção documenta **o que é necessário** para reproduzir e justificar a segmentação dos pareceres antes da indexação. A implementação está desenvolvida em `src/pipeline.py` (função `run_ingest` e auxiliares), com parâmetros em `config.yaml` → `legal_chunking` e `transform`.

### 10.1 Entrada e pré-requisitos

1. **Markdown por documento** em `data/transformation/`, listado na transformação (etapa `--step transform`). Cada ficheiro pode incluir **front matter YAML** inicial (`---` … `---`) com metadados (`doc_id`, `titulo`, `fonte`, `data`, `tipo`, `assunto`).
2. **Normalização de estrutura** (opcional, recomendada): com `transform.normalize_legal_headings: true`, os títulos típicos de parecer (EMENTA, RELATÓRIO, FUNDAMENTAÇÃO, CONCLUSÃO, DISPOSITIVO, numeração romana, etc.) são padronizados como headings `##` antes da ingestão, o que **melhora a detecção de seções** descrita abaixo.

### 10.2 Primeiro nível: seções jurídicas

O corpo do Markdown (após remoção do front matter) é percorrido **linha a linha** (`_detect_legal_sections`):

- **Padrões de cabeçalho** alinhados a peças do Estado de Goiás: EMENTA, RELATÓRIO, FUNDAMENTAÇÃO, CONCLUSÃO, DISPOSITIVO (incluindo variantes com/sem acentos e com numeração tipo `I - RELATÓRIO`).
- **Qualquer linha que comece por `#`** inicia nova seção; o texto acumulado até ao próximo limiar é guardado como um bloco.
- Cada bloco recebe um rótulo (`ementa`, `relatorio`, `fundamentacao`, `conclusao`, `dispositivo` ou `secao_geral`) para fins de **filtro semântico e de metadados** no RAG.

**Justificativa:** pareceres tributários seguem uma **ordem argumentativa estável**; segmentar por estas seções evita misturar, num único chunk longo, relatório fático, fundamentação jurídica e conclusão, o que prejudicaria tanto a recuperação por embedding como a explicação da citação ao utilizador.

### 10.3 Segundo nível: sub-chunks por tamanho (tokens)

Seções longas são ainda divididas (`_split_section_into_subchunks`):

- **Contagem de tokens:** usa `nltk.word_tokenize` quando disponível; caso contrário, **fallback** por tokens alfanuméricos (`\w+`).
- **`legal_chunking.max_tokens`** (no projeto: **700**): tamanho máximo da janela.
- **`legal_chunking.overlap_tokens`** (no projeto: **140**): sobreposição entre janelas consecutivas; o passo efectivo é `max_tokens - overlap_tokens` (mínimo 1), gerando **janela deslizante** sobre a sequência de tokens.

**Nota técnica:** a divisão é **por token**, não por parágrafos nem por `RecursiveCharacterTextSplitter`; prioriza **limite de contexto** e reprodutibilidade em texto jurídico denso. Listas e tabelas herdadas do Markdown ficam no fluxo linear da seção — podem ser cortadas a meio da janela; isso é uma **limitação aceita** face à necessidade de chunks de tamanho controlado para o vetor e para o LLM.

### 10.4 Identificação, metadados e texto enriquecido

- **`chunk_id`:** `{doc_id}#chunk_` + índice sequencial de quatro dígitos por documento (`0000`, `0001`, …).
- **Extras extraídos do texto bruto do sub-chunk:** `normas_citadas` (regex sobre artigos, leis, CTN, CF/88, etc.) e `tributos_citados` (keywords: ICMS, IPVA, ITCD, ISS, …).
- **Texto indexado no denso (`texto`):** versão **enriquecida** (`_build_enriched_text`) com prefixo explícito: tipo de documento, `doc_id`, título, assunto, seção, normas e tributos agregados, seguido do texto. Isto **reforça o sinal semântico** do embedding com o contexto normativo. O **BM25** e o contexto apresentado ao LLM podem usar `texto_bruto` / metadados conforme o fluxo em `rag_core.py`.

### 10.5 Saída e comando

- **Ficheiro gerado:** `data/processed/chunks.json` (lista de objetos com `chunk_id`, `doc_id`, `texto`, `texto_bruto`, `metadados`).
- **Comando:** `python src/pipeline.py --step ingest` (após `transform`).
