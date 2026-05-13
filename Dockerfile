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

# LTO desabilitado: GCC 13 lança ICE (internal compiler error) ao linkar com LTO
# sob emulação QEMU x86-64 (Mac ARM64 host). -march=x86-64-v3 garante AVX2+BMI2.
RUN mkdir -p build && cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -fno-rtti -funroll-loops" \
        .. && \
    make -j$(nproc)

ENV RESOURCES_DIR=/app/resources
RUN ./build/vector_based_fraud_detection_api --prepare

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
