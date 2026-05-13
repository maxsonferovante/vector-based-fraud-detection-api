# Stage 1: Build
# --platform linux/amd64: garante compilação para x86-64 independente do host (Mac ARM64).
FROM --platform=linux/amd64 ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

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
COPY . .

# ── Build Pass 1: PGO instrumentation ──
# Compila com -fprofile-generate para coletar perfil de execução.
RUN mkdir -p build-pgo && cd build-pgo && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -fno-rtti -funroll-loops" \
        -DPGO_GENERATE=ON \
        .. && \
    make -j$(nproc)

# Gera o índice binário E coleta perfil PGO durante o processamento.
ENV RESOURCES_DIR=/app/resources
RUN cd build-pgo && ./vector_based_fraud_detection_api --prepare

# ── Build Pass 2: PGO otimizado + LTO ──
# Reconstrói usando o perfil coletado para otimizar hot paths.
RUN mkdir -p build && cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCPP_LTO=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -fno-rtti -funroll-loops" \
        -DPGO_USE=ON \
        -DPGO_DIR=/app/build-pgo/pgo-data \
        .. && \
    make -j$(nproc)

# O índice binário já foi gerado no pass 1; reutilizar.
RUN cp /app/build-pgo/resources/matcher.bin /app/resources/matcher.bin 2>/dev/null || true

# Stage 2: Runtime
FROM --platform=linux/amd64 ubuntu:24.04

RUN apt-get update && apt-get install -y \
    libboost-system1.83.0 \
    libgomp1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/vector_based_fraud_detection_api .
COPY --from=builder /app/resources/ /app/resources/

EXPOSE 9999

CMD ["./vector_based_fraud_detection_api"]
