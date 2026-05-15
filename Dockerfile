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
# CORREÇÃO: adicionamos -mavx2 explicitamente para garantir que o código SIMD
# seja gerado e perfilado — sem isso, o GCC pode usar um caminho escalar no
# pass 1 e então aplicar as otimizações PGO no caminho errado no pass 2.
RUN mkdir -p build-pgo && cd build-pgo && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CXX_FLAGS="-O2 -march=x86-64-v3 -mavx2 -funroll-loops" \
        -DPGO_GENERATE=ON && \
    ninja -j2

# ── Fase --prepare: constrói índice E coleta perfil PGO do hot path real ──
#
# CORREÇÃO: o --prepare agora faz duas coisas:
#   1. Constrói e salva o índice binário (matcher.bin)
#   2. Executa 10.000 queries sintéticas de warm-up via run_pgo_warmup()
#      para que o compilador colete perfil do search(), partial_sort dos
#      centroides, loop SIMD de distância e bail de norma — o código que
#      realmente executa em produção.
#
# Antes, o perfil coletado era apenas do k-means e do save_binary(),
# que não executam em produção. O PGO era essencialmente inútil.
ENV RESOURCES_DIR=/app/resources
RUN cd build-pgo && ./vector_based_fraud_detection_api --prepare

# ── Build Pass 2: PGO otimizado + LTO ──
# Reconstrói usando o perfil coletado para otimizar hot paths.
# O uso do Ninja evita o bug LTO jobserver do GCC.
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCPP_LTO=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -mavx2 -funroll-loops" \
        -DPGO_USE=ON \
        -DPGO_DIR=/app/build-pgo/pgo-data && \
    ninja -j2

# Reutiliza o índice binário gerado no pass 1.
# NOTA: se NUM_CLUSTERS ou KMEANS_ITERS mudarem no código, o INDEX_VERSION
# no simd_ivf_matcher.hpp deve ser incrementado — o load_binary() vai
# rejeitar o índice antigo e fazer fallback para o .gz, forçando rebuild.
RUN cp /app/build-pgo/resources/matcher.bin /app/resources/matcher.bin 2>/dev/null || true

# Stage 2: Runtime — imagem mínima sem toolchain de compilação
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