#!/bin/bash

# Este script compila a imagem Docker para a arquitetura x86_64 (amd64) 
# exigida pela Rinha de Backend e a envia para o Docker Hub.

IMAGE_NAME="maxsonferovante/vector-based-fraud-detection-api:latest"

echo "========================================================="
echo "Iniciando o build cruzado para a plataforma linux/amd64..."
echo "Imagem alvo: $IMAGE_NAME"
echo "========================================================="

# Utiliza o buildx para garantir que a imagem seja amd64 independente 
# de estar rodando em um Mac com Apple Silicon (ARM) ou Windows/Linux.
docker buildx build --platform linux/amd64 -t "$IMAGE_NAME" --push .

if [ $? -eq 0 ]; then
    echo "========================================================="
    echo "Sucesso! A imagem foi compilada (amd64) e enviada para o Docker Hub."
    echo "========================================================="
else
    echo "========================================================="
    echo "Erro! Houve um problema durante o build ou o push."
    echo "Certifique-se de que o Docker Desktop está rodando e autenticado."
    echo "========================================================="
    exit 1
fi
