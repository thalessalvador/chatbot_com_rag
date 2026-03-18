import pandas as pd
from difflib import SequenceMatcher

def calculate_similarity(a, b):
    """Retorna a razão de similaridade entre duas strings (0.0 a 1.0)."""
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()

# 1. Carregar o arquivo original
input_file = 'log_proxy_202603151043.csv'
df = pd.read_csv(input_file, sep=';', encoding='latin-1')

# 2. Processar a deduplicação
indices_to_keep = []  # Armazena índices das linhas únicas
kept_questions = []   # Armazena o texto das perguntas já mantidas
threshold = 0.80      # Limite de 80% de similaridade

for index, row in df.iterrows():
    current_q = row['pergunta']
    is_duplicate = False
    
    # Compara a pergunta atual com todas que já decidimos manter
    for kept_q in kept_questions:
        if calculate_similarity(current_q, kept_q) >= threshold:
            is_duplicate = True
            break
            
    if not is_duplicate:
        indices_to_keep.append(index)
        kept_questions.append(current_q)

# 3. Criar o novo DataFrame e salvar
df_cleaned = df.iloc[indices_to_keep].copy()
output_file = 'log_proxy_deduplicado.csv'
df_cleaned.to_csv(output_file, sep=';', index=False)

print(f"Processamento concluído:")
print(f"- Total de registros originais: {len(df)}")
print(f"- Registros após limpeza: {len(df_cleaned)}")
print(f"- Duplicadas removidas: {len(df) - len(df_cleaned)}")