# Usar uma imagem base oficial do Python
FROM python:3.11-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar o arquivo de dependências primeiro (para otimizar o cache)
COPY requirements.txt requirements.txt

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto do código da aplicação
COPY . .

# Expor a porta que o Flask usa
EXPOSE 5000

# Comando para iniciar a aplicação quando o contêiner rodar
CMD ["python", "app.py"]