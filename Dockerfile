# Usa uma imagem oficial do Python, versão slim para ser mais leve
FROM python:3.10-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Instala dependências do sistema necessárias para compilação (ex: ChromaDB, BM25)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Cria diretórios necessários para os volumes
RUN mkdir -p data/raw data/processed data/index

# Copia o restante do código do projeto
COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501