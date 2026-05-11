#!/bin/bash

# Este script compila a imagem Docker para a arquitetura x86_64 (amd64) 
# exigida pela Rinha de Backend e a envia para o Docker Hub com versionamento de data/hora.

# Gera a tag no formato dd-mm-aaaa-hh-mm-ss
TIMESTAMP=$(date +"%d-%m-%Y-%H-%M-%S")
IMAGE_NAME="maxsonferovante/vector-based-fraud-detection-api"
IMAGE_TAGGED="$IMAGE_NAME:$TIMESTAMP"
IMAGE_LATEST="$IMAGE_NAME:latest"

echo "========================================================="
echo "Iniciando o build cruzado para a plataforma linux/amd64..."
echo "Imagem alvo: $IMAGE_TAGGED"
echo "Também atualizando a tag 'latest'"
echo "========================================================="

# Utiliza o buildx para garantir que a imagem seja amd64. 
# Faz o build e tagueia com o timestamp e também como latest.
docker buildx build --platform linux/amd64 \
  -t "$IMAGE_TAGGED" \
  -t "$IMAGE_LATEST" \
  --push .

if [ $? -eq 0 ]; then
    echo "========================================================="
    echo "Sucesso! A imagem foi compilada (amd64) e enviada para o Docker Hub."
    echo "Tags enviadas: $TIMESTAMP e latest"
    echo "========================================================="
else
    echo "========================================================="
    echo "Erro! Houve um problema durante o build ou o push."
    echo "Certifique-se de que o Docker Desktop está rodando e autenticado."
    echo "========================================================="
    exit 1
fi
