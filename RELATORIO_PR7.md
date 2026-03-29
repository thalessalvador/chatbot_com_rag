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

## 4. Baseline Confirmada

Baseline **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`): **Recall@k** por modo de recuperacao, com os **29** casos do golden set utilizados na consolidacao de `docs/dados/recall_por_k.json` (actualizado em 2026-03-29).

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

Resumo da linha **k = 5** (golden set alargado face a corridas anteriores com 26 itens):

- Modo Denso: Recall@5 de **41,38%** (12/29 acertos)
- Modo Esparso: Recall@5 de **68,97%** (20/29 acertos)
- Modo Hibrido: Recall@5 de **72,41%** (21/29 acertos)

## 5. Experimento com Embedding Juridico

Tabela consolidada de **Recall@k** (denso, esparso, hibrido) para o baseline **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) e o embedding **juridico** (`stjiris/bert-large-portuguese-cased-legal-tsdae`). Valores obtidos via `run_evaluate` e registados em `docs/dados/recall_por_k.json` (**29** perguntas com gabarito utilizavel nesta execucao).

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

Foi realizado um experimento controlado alterando apenas o modelo de embeddings para:

- `stjiris/bert-large-portuguese-cased-legal-tsdae`

Resultado obtido (linha **k = 5** da tabela acima, BERT legal):

- Modo Denso: Recall@5 de **17,24%** (5/29 acertos)
- Modo Esparso: Recall@5 de **68,97%** (20/29 acertos)
- Modo Hibrido: Recall@5 de **72,41%** (21/29 acertos)

## 6. Conclusao Tecnica

Para avaliacao do modelo foi construído um script capaz de rodar ambos os modelos em k-posições necessárias. Para tal, rode (para k IN (3, 5 e 10)):

```powershell
python scripts/run_recall_at_k_values.py --k-values 3,5,10
```

![Recall@3 — MiniLM vs BERT legal](docs/imagens/recall_k3.png)

![Recall@5 — MiniLM vs BERT legal](docs/imagens/recall_k5.png)

![Recall@10 — MiniLM vs BERT legal](docs/imagens/recall_k10.png)

Com o golden set **atual (29 perguntas validas)**, o embedding juridico **mantem desvantagem clara no modo denso** face ao MiniLM (por exemplo, Recall@5 de **17,24%** frente a **41,38%**). No modo **esparso**, os dois modelos **empatam** em todos os k reportados (mesmos acertos/total, por o BM25 nao depender do embedding). No modo **hibrido**, para **Recall@5** os dois embeddings produzem **o mesmo resultado: 72,41%** (21/29); para **Recall@10**, o **MiniLM** fica ligeiramente **acima** (**82,76%** ou 24/29) do BERT legal (**79,31%** ou 23/29).

Em sintese, apos o alargamento do conjunto de testes, o ganho exclusivo do BERT no hibrido observado na versao anterior do relatorio **deixa de aparecer em k = 5** (empate). A **Trilha A** continua relevante porque o **hibrido** supera denso e esparso isolados em varios cenarios; a escolha do embedding deve equilibrar **custo/tempo** (BERT maior) e o facto de, neste conjunto, o **MiniLM** ainda ser **superior ou igual** ao BERT no denso e no hibrido em **k = 10**.

## 7. Implementacao na Interface

A interface em Streamlit foi desenhada para espelhar estas escolhas tecnicas no contacto com o utilizador. Na barra lateral, e possivel alternar entre **Hibrido**, **Denso** e **Esparso**, de modo que a mesma pergunta possa ser respondida com estrategias distintas sem sair da aplicacao. Isto aproxima a demonstracao oral do que o relatório quantifica: o utilizador ve a resposta com citacoes, pode abrir o painel de trechos recuperados e, ao mudar o modo, observa diretamente como mudam os chunks que sustentam o RAG.
