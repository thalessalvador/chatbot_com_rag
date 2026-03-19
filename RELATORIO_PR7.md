
# Relatório de Validação e Avaliação do RAG (PR5, PR6 e PR7)

## 1. Visão Geral

Este documento apresenta os resultados da avaliação formal do sistema de Retrieval-Augmented Generation (RAG) para Pareceres Tributários da SEFAZ-GO. O objetivo desta fase foi medir matematicamente a precisão da recuperação de informação (PR5) e expor as opções tecnológicas na interface final (PR6).

## 2. Artefactos Produzidos (Golden Set)

Foi consolidado um **Golden Set** composto por 27 perguntas reais de negócio formuladas por especialistas (Auditores).

Para permitir a avaliação empírica, foi criada uma rotina de _Bootstrapping_ que cruzou as "Respostas Esperadas" com a base vetorial, identificando automaticamente o Documento/Parecer de origem correto para cada uma das 27 perguntas, servindo como "Gabarito" (`data/Golden_Set_Preenchido_pelo_RAG.xlsx`).

## 3. Avaliação Formal (PR5 - Recall@5)

Para avaliar a qualidade do motor de pesquisa, utilizámos a métrica **Recall@5** (percentagem de vezes que o documento correto apareceu entre os 5 primeiros resultados). Foram testadas três estratégias diferentes:

-   **Denso (Embeddings):** Pesquisa semântica usando o modelo `all-MiniLM-L6-v2` (384 dimensões).
    
-   **Esparso (BM25):** Pesquisa lexical clássica baseada na frequência exata de termos e palavras-chave.
    
-   **Híbrido (RRF):** Combinação matemática de ambos os métodos utilizando a técnica _Reciprocal Rank Fusion_.
    

### Resultados Obtidos:

-   **Modo Denso:** Recall@5 de **[18.5]%**
    
-   **Modo Esparso:** Recall@5 de **[11.1]%**
    
-   **Modo Híbrido:** Recall@5 de **[22.2]%**
    

### Conclusão Técnica:

A estratégia híbrida demonstrou ser a mais eficaz. Ela compensou a falta de compreensão de contexto do BM25 (esparso) e as eventuais falhas do modelo denso em identificar códigos de leis e artigos exatos, garantindo a maior taxa de acerto e consistência do projeto.

## 4. Implementação na Interface (PR6)

Com base na robustez arquitetural desenvolvida na camada `rag_core`, a interface do Streamlit (`app.py`) foi atualizada. Foi adicionado um seletor na barra lateral que expõe, de forma transparente, as 3 estratégias de pesquisa ao utilizador final, permitindo flexibilidade em consultas muito complexas.

_Artefactos e métricas validados e integrados no repositório principal._