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
    libopenblas-dev \
    liblapack-dev \
    libgomp1 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /deps

# 1. Build FAISS from source (v1.8.0 stable)
RUN git clone --depth 1 --branch v1.8.0 https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build \
          -DFAISS_ENABLE_GPU=OFF \
          -DFAISS_ENABLE_PYTHON=OFF \
          -DBUILD_TESTING=OFF \
          -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_BUILD_TYPE=Release . && \
    cmake --build build -j$(nproc) && \
    cmake --install build

WORKDIR /app

# Copia o código fonte (respeitando .dockerignore)
COPY . .

# Compila a aplicação
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# ==========================================
# STAGE 2: Runtime
# ==========================================
FROM ubuntu:24.04

# Instala apenas as bibliotecas de runtime necessárias
RUN apt-get update && apt-get install -y \
    libboost-system1.83.0 \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o binário e os recursos do builder
COPY --from=builder /app/build/vector_based_fraud_detection_api .
COPY --from=builder /usr/local/lib/libfaiss.so* /usr/local/lib/
COPY resources/ /app/resources/

# Atualiza cache de bibliotecas
RUN ldconfig

EXPOSE 9999

CMD ["./vector_based_fraud_detection_api"]
