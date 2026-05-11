# ==========================================
# STAGE 1: Build
# ==========================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Instala ferramentas de build e dependências básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libboost-system-dev \
    libgomp1 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o código fonte (respeitando .dockerignore)
COPY . .

# Compila a aplicação
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Gera o binário SIMD otimizado durante o build
ENV RESOURCES_DIR=/app/resources
RUN ./build/vector_based_fraud_detection_api --prepare

# ==========================================
# STAGE 2: Runtime
# ==========================================
FROM ubuntu:24.04

# Instala apenas as bibliotecas de runtime necessárias
RUN apt-get update && apt-get install -y \
    libboost-system1.83.0 \
    libgomp1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o binário e os recursos do builder (contendo matcher.bin)
COPY --from=builder /app/build/vector_based_fraud_detection_api .
COPY --from=builder /app/resources/ /app/resources/

EXPOSE 9999

CMD ["./vector_based_fraud_detection_api"]
