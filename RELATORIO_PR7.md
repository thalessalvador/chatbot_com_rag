# Relatorio de Validacao e Avaliacao do RAG (PR5, PR6 e PR7)

## 1. Visao Geral

Este documento apresenta os resultados da avaliacao formal do sistema de Retrieval-Augmented Generation (RAG) para Pareceres Tributarios da SEFAZ-GO. O objetivo desta fase foi medir matematicamente a qualidade da recuperacao de informacao (PR5), consolidar a exposicao dos modos de busca na interface (PR6) e registrar os artefatos finais da etapa experimental (PR7).

## 2. Artefatos Produzidos

Foi utilizado um Golden Set em Excel com perguntas de negocio e documento esperado, consumido diretamente pela etapa:

```powershell
python src/pipeline.py --step evaluate --golden-file Golden_Set_Preenchido_pelo_RAG_Reranked.xlsx --k 5
```

Na execucao validada nesta fase, a avaliacao considerou 25 perguntas com gabarito utilizavel.

## 3. Estrategias Avaliadas

Foram comparadas tres estrategias de recuperacao:

- Denso (embeddings): recuperacao semantica por vetores.
- Esparso (BM25): recuperacao lexical por frequencia exata de termos.
- Hibrido (RRF): combinacao entre denso e esparso por Reciprocal Rank Fusion.

## 4. Baseline Confirmada

A baseline revalidada com o modelo `all-MiniLM-L6-v2` produziu:

- Modo Denso: Recall@5 de 24,0% (6/25 acertos)
- Modo Esparso: Recall@5 de 56,0% (14/25 acertos)
- Modo Hibrido: Recall@5 de 56,0% (14/25 acertos)

Esses valores substituem os numeros anteriormente anotados neste relatorio, que nao correspondiam ao mesmo cenario experimental hoje reprodutivel via linha de comando.

## 5. Experimento com Embedding Juridico

Foi realizado um experimento controlado alterando apenas o modelo de embeddings para:

- `stjiris/bert-large-portuguese-cased-legal-tsdae`

Resultado obtido:

- Modo Denso: Recall@5 de 12,0% (3/25 acertos)
- Modo Esparso: Recall@5 de 56,0% (14/25 acertos)
- Modo Hibrido: Recall@5 de 64,0% (16/25 acertos)

## 6. Conclusao Tecnica

O experimento mostrou um comportamento relevante:

- O novo embedding piorou a busca densa isolada.
- O modo esparso permaneceu estavel.
- O modo hibrido melhorou de 56,0% para 64,0%.

Portanto, embora o embedding `stjiris/bert-large-portuguese-cased-legal-tsdae` nao seja superior ao baseline no modo denso, ele melhorou a recuperacao final no modo hibrido, que e o modo mais importante para o projeto.

## 7. Implementacao na Interface

A interface Streamlit expoe os tres modos de busca ao utilizador final:

- Hibrido
- Denso
- Esparso

Isso permite inspecao pratica dos comportamentos do retriever e alinhamento entre avaliacao formal e uso interativo.
