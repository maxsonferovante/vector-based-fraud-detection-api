# Stage 1: Build
# --platform linux/amd64: garante compilação para x86-64 independente do host (Mac ARM64).
FROM --platform=linux/amd64 alpine:3.20 AS builder

RUN apk add --no-cache \
    build-base \
    cmake \
    ninja \
    git \
    linux-headers \
    musl-dev \
    zlib-dev \
    boost-dev

WORKDIR /app
COPY . .

# ── Build Pass 1: PGO instrumentation ──
# Compila com -fprofile-generate para coletar perfil de execução.
RUN mkdir -p build-pgo && cd build-pgo && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -funroll-loops" \
        -DPGO_GENERATE=ON && \
    ninja -j"$(nproc)"

# Gera o índice binário E coleta perfil PGO durante o processamento.
ENV RESOURCES_DIR=/app/resources
RUN cd build-pgo && ./vector_based_fraud_detection_api --prepare

# ── Build Pass 2: PGO otimizado + LTO ──
# Reconstrói usando o perfil coletado para otimizar hot paths.
# O uso do Ninja evita o bug LTO jobserver do GCC.
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCPP_LTO=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -funroll-loops" \
        -DPGO_USE=ON \
        -DPGO_DIR=/app/build-pgo/pgo-data && \
    ninja -j"$(nproc)"

# O índice binário já foi gerado no pass 1; reutilizar.
RUN cp /app/build-pgo/resources/matcher.bin /app/resources/matcher.bin 2>/dev/null || true

# Stage 2: Runtime
FROM --platform=linux/amd64 alpine:3.20

RUN apk add --no-cache \
    libstdc++ \
    zlib \
    libgcc

WORKDIR /app

COPY --from=builder /app/build/vector_based_fraud_detection_api .
COPY --from=builder /app/resources/ /app/resources/

EXPOSE 9999

CMD ["./vector_based_fraud_detection_api"]
