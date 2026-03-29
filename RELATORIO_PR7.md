# Relatorio de Validacao e Avaliacao do RAG (PR5, PR6 e PR7)

## 1. Visao Geral

Este documento apresenta os resultados da avaliacao formal do sistema de Retrieval-Augmented Generation (RAG) para Pareceres Tributarios da SEFAZ-GO. O objetivo desta fase foi medir matematicamente a qualidade da recuperacao de informacao (PR5), consolidar a exposicao dos modos de busca na interface (PR6) e registrar os artefatos finais da etapa experimental (PR7).

## 2. Artefatos Produzidos

Foi utilizado um Golden Set em Excel com perguntas de negocio e documento esperado, consumido diretamente pela etapa:

```powershell
python src/pipeline.py --step evaluate --golden-file Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx --k 5
```

Na execucao validada nesta fase, a avaliacao considerou **26** perguntas com gabarito utilizavel (golden set actual).

## 3. Estrategias Avaliadas

Foram comparadas tres estrategias de recuperacao:

- Denso (embeddings): recuperacao semantica por vetores.
- Esparso (BM25): recuperacao lexical por frequencia exata de termos.
- Hibrido (RRF): combinacao entre denso e esparso por Reciprocal Rank Fusion.

## 4. Baseline Confirmada

Baseline **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`): **Recall@k** por modo de recuperacao, com os mesmos 26 casos do golden set utilizados na consolidacao de `docs/dados/recall_por_k.json`.

| k | Modo | Recall (%) | acertos/total |
|---|------|------------|---------------|
| 3 | Denso | 23,08 | 6/26 |
| 3 | Esparso | 50,00 | 13/26 |
| 3 | Hibrido | 53,85 | 14/26 |
| 5 | Denso | 30,77 | 8/26 |
| 5 | Esparso | 53,85 | 14/26 |
| 5 | Hibrido | 57,69 | 15/26 |
| 10 | Denso | 34,62 | 9/26 |
| 10 | Esparso | 53,85 | 14/26 |
| 10 | Hibrido | 69,23 | 18/26 |

Resumo da linha **k = 5** (comparavel a corridas anteriores do projeto com 25 itens, hoje com 26):

- Modo Denso: Recall@5 de **30,77%** (8/26 acertos)
- Modo Esparso: Recall@5 de **53,85%** (14/26 acertos)
- Modo Hibrido: Recall@5 de **57,69%** (15/26 acertos)

## 5. Experimento com Embedding Juridico

Tabela consolidada de **Recall@k** (denso, esparso, hibrido) para o baseline **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) e o embedding **juridico** (`stjiris/bert-large-portuguese-cased-legal-tsdae`). Valores obtidos via `run_evaluate` e registados em `docs/dados/recall_por_k.json` (26 perguntas do golden set com gabarito utilizavel nesta execucao).

| k | Modo | MiniLM — Recall (%) | MiniLM — acertos/total | BERT legal — Recall (%) | BERT legal — acertos/total |
|---|------|---------------------|------------------------|-------------------------|----------------------------|
| 3 | Denso | 23,08 | 6/26 | 19,23 | 5/26 |
| 3 | Esparso | 50,00 | 13/26 | 50,00 | 13/26 |
| 3 | Hibrido | 53,85 | 14/26 | 61,54 | 16/26 |
| 5 | Denso | 30,77 | 8/26 | 19,23 | 5/26 |
| 5 | Esparso | 53,85 | 14/26 | 53,85 | 14/26 |
| 5 | Hibrido | 57,69 | 15/26 | 65,38 | 17/26 |
| 10 | Denso | 34,62 | 9/26 | 23,08 | 6/26 |
| 10 | Esparso | 53,85 | 14/26 | 53,85 | 14/26 |
| 10 | Hibrido | 69,23 | 18/26 | 65,38 | 17/26 |

Foi realizado um experimento controlado alterando apenas o modelo de embeddings para:

- `stjiris/bert-large-portuguese-cased-legal-tsdae`

Resultado obtido (linha **k = 5** da tabela acima, BERT legal):

- Modo Denso: Recall@5 de 19,23% (5/26 acertos)
- Modo Esparso: Recall@5 de 53,85% (14/26 acertos)
- Modo Hibrido: Recall@5 de 65,38% (17/26 acertos)

## 6. Conclusao Tecnica

Para avaliação do modelo foi construído um script capaz de rodar ambos os modelos em k-posições necessárias. Para tal, rode (para k IN (3, 5 e 10)):

```powershell
python scripts/run_recall_at_k_values.py --k-values 3,5,10
```

![Recall@3 — MiniLM vs BERT legal](docs/imagens/recall_k3.png)

![Recall@5 — MiniLM vs BERT legal](docs/imagens/recall_k5.png)

![Recall@10 — MiniLM vs BERT legal](docs/imagens/recall_k10.png)

Em sintese, a troca do embedding generico pelo modelo juridico especializado **não** trouxe vantagem no modo **denso isolado**, onde o recall ate regrediu. O modo **esparso** houve uma ligeira melhora, indicando que o modelo especialista ranqueia melhor as respostas, mas indica apenas a mudança de apenas 1 pergunta do Golden Set . O modo **hibrido**, por sua vez, elevou o Recall@5 de **57,69%** (MiniLM) para **65,38%** (BERT legal), consolidando a Trilha A como mais do que uma combinacao formal: neste conjunto de testes, foi o unico cenario em que o embedding juridico se traduziu em **ganho mensuravel** na recuperacao. Assim, para o uso previsto do chatbot — com fusao densa-esparsa ativa — o modelo `stjiris/bert-large-portuguese-cased-legal-tsdae` mostrou-se pertinente.

## 7. Implementacao na Interface

A interface em Streamlit foi desenhada para espelhar estas escolhas tecnicas no contacto com o utilizador. Na barra lateral, e possivel alternar entre **Hibrido**, **Denso** e **Esparso**, de modo que a mesma pergunta possa ser respondida com estrategias distintas sem sair da aplicacao. Isto aproxima a demonstracao oral do que o relatório quantifica: o utilizador ve a resposta com citacoes, pode abrir o painel de trechos recuperados e, ao mudar o modo, observa diretamente como mudam os chunks que sustentam o RAG.
