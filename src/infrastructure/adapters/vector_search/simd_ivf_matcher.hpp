#ifndef SIMD_IVF_MATCHER_HPP
#define SIMD_IVF_MATCHER_HPP

#include "../../../application/ports/out/vector_search_port.hpp"
#include "../../logging/logger.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <numeric>
#include <cstdint>
#include <cstdio>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

#ifdef __linux__
    #include <sys/mman.h>
#endif

namespace infrastructure {
namespace adapters {
namespace vector_search {

// IVF (Inverted File Index) com busca SIMD AVX2/NEON.
// Otimizações v2: norm pruning, prefetch, rounding quantization, hugepages.
// Índice construído offline via train_and_build (--prepare);
// search() é read-only e thread-safe após a construção.
class SimdIvfMatcher : public application::ports::out::VectorSearchPort {
public:
    // Norm pruning: uint16_t norm pré-computado permite lower-bound check via
    // desigualdade triangular, evitando ~50-80% dos cálculos SIMD de distância.
    // Buckets são ordenados por norm para early termination (bail_norm).
    // sizeof = 36 bytes → 3M × 36 = ~103 MB (dentro do budget de 150 MB).
    struct PaddedVector {
        int16_t elements[16];   // 32 bytes — vetor quantizado (14 dims + 2 padding)
        uint16_t norm;          //  2 bytes — L2 norm pré-computada (ceil(sqrt(sum_sq)))
        bool is_fraud;          //  1 byte
        // padding:                1 byte  (alinhamento a 2 bytes)
    };

private:
    static constexpr size_t   NUM_CLUSTERS   = 2048;
    static constexpr size_t   PROBE_CLUSTERS = 16;
    static constexpr float    SCALE          = 4096.0f;
    static constexpr int      KMEANS_ITERS   = 3;
    static constexpr size_t   PREFETCH_AHEAD = 16;  // prefetch 16 vetores à frente

    struct alignas(32) Centroid {
        int16_t elements[16];
    };

    std::vector<Centroid>                   centroids_;
    std::vector<std::vector<PaddedVector>>  buckets_;
    size_t total_vectors_ = 0;

    // --- Quantização com arredondamento ---
    // Reduz bias de truncamento: +0.5f para positivos, -0.5f para negativos (sentinela -1).
    static inline int16_t quantize(float v) noexcept {
        float scaled = v * SCALE;
        return (v >= 0.0f)
            ? static_cast<int16_t>(scaled + 0.5f)
            : static_cast<int16_t>(scaled - 0.5f);
    }

    // --- Norm computation ---
    // Retorna ceil(sqrt(sum of squares)) das 14 dimensões quantizadas.
    // uint16_t suficiente: max = ceil(sqrt(14 * 4096^2)) = 15327.
    static inline uint16_t compute_norm(const int16_t* v) noexcept {
        int32_t sum = 0;
        for (int i = 0; i < 14; ++i) {
            int32_t val = static_cast<int32_t>(v[i]);
            sum += val * val;
        }
        double s = std::sqrt(static_cast<double>(sum));
        auto up = static_cast<uint16_t>(s);
        if (static_cast<double>(up) < s) ++up;
        return up;
    }

    // --- Bail norm ---
    // Limite superior para early termination em buckets ordenados por norm.
    // Se norm_ref >= bail_norm, ||q - ref||^2 >= threshold (pela desigualdade triangular).
    static inline uint32_t compute_bail_norm(uint32_t q_norm, int32_t threshold) noexcept {
        if (threshold == INT32_MAX) return UINT32_MAX;
        double s = std::sqrt(static_cast<double>(threshold));
        auto up = static_cast<uint32_t>(s);
        if (static_cast<double>(up) < s) ++up;
        uint64_t b = static_cast<uint64_t>(q_norm) + up + 1u;
        return (b > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(b);
    }

    // simd_dist: distância euclidiana quadrada para ponteiros 32-byte aligned (Centroid).
    static inline int32_t simd_dist(const int16_t* __restrict__ a,
                                    const int16_t* __restrict__ b) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        __m256i va  = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
        __m256i vb  = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
        __m256i d   = _mm256_sub_epi16(va, vb);
        __m256i sq  = _mm256_madd_epi16(d, d);     // horizontal multiply-add pairwise → int32
        __m128i lo  = _mm256_castsi256_si128(sq);
        __m128i hi  = _mm256_extracti128_si256(sq, 1);
        __m128i s   = _mm_add_epi32(lo, hi);
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1,0,3,2)));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1)));
        return _mm_cvtsi128_si32(s);
#elif defined(__aarch64__) || defined(_M_ARM64)
        int16x8_t va_lo = vld1q_s16(a);
        int16x8_t va_hi = vld1q_s16(a + 8);
        int16x8_t vb_lo = vld1q_s16(b);
        int16x8_t vb_hi = vld1q_s16(b + 8);
        int16x8_t dl    = vsubq_s16(va_lo, vb_lo);
        int16x8_t dh    = vsubq_s16(va_hi, vb_hi);
        int32x4_t ml    = vmull_s16(vget_low_s16(dl), vget_low_s16(dl));
        ml = vmlal_s16(ml, vget_high_s16(dl), vget_high_s16(dl));
        int32x4_t mh    = vmull_s16(vget_low_s16(dh), vget_low_s16(dh));
        mh = vmlal_s16(mh, vget_high_s16(dh), vget_high_s16(dh));
        return vaddvq_s32(vaddq_s32(ml, mh));
#else
        int32_t dist = 0;
        for (int j = 0; j < 14; ++j) {
            int32_t d = (int32_t)a[j] - (int32_t)b[j];
            dist += d * d;
        }
        return dist;
#endif
    }

    // simd_dist_pv: variante com loadu para PaddedVector (sem alignas → sem garantia de alinhamento).
    static inline int32_t simd_dist_pv(const int16_t* __restrict__ a,
                                       const int16_t* __restrict__ b) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        __m256i va  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        __m256i vb  = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
        __m256i d   = _mm256_sub_epi16(va, vb);
        __m256i sq  = _mm256_madd_epi16(d, d);
        __m128i lo  = _mm256_castsi256_si128(sq);
        __m128i hi  = _mm256_extracti128_si256(sq, 1);
        __m128i s   = _mm_add_epi32(lo, hi);
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1,0,3,2)));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1)));
        return _mm_cvtsi128_si32(s);
#else
        return simd_dist(a, b); // scalar e NEON: loadu e load são equivalentes
#endif
    }

public:
    SimdIvfMatcher() {
        buckets_.resize(NUM_CLUSTERS);
        for (auto& b : buckets_) b.reserve(2000);
    }

    // Constrói o índice IVF com k-means iterativo.
    // Executado apenas no --prepare (offline); impacto zero no runtime.
    // Fluxo: random seed → KMEANS_ITERS x (assignment SIMD + update int64) → fill buckets → sort by norm.
    void train_and_build(const std::vector<std::pair<std::array<float, 14>, bool>>& data) {
        if (data.empty()) return;

        const size_t N = data.size();
        const size_t K = std::min(NUM_CLUSTERS, N);

        logging::Logger::info("Building IVF index: k-means iters=" +
                              std::to_string(KMEANS_ITERS) + " clusters=" +
                              std::to_string(K));

        // Converte float→int16 com arredondamento; alinhado a 32 bytes para loads SIMD.
        struct alignas(32) AlignedVec { int16_t e[16]; };
        std::vector<AlignedVec> vecs(N);
        for (size_t i = 0; i < N; ++i) {
            const auto& v = data[i].first;
            for (int j = 0; j < 14; ++j)
                vecs[i].e[j] = quantize(v[j]);
            vecs[i].e[14] = 0; vecs[i].e[15] = 0;
        }

        // Semente: amostragem aleatória com seed fixo para reprodutibilidade.
        std::mt19937 rng(42);
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        centroids_.resize(K);
        for (size_t i = 0; i < K; ++i)
            std::copy(vecs[indices[i]].e, vecs[indices[i]].e + 16, centroids_[i].elements);

        std::vector<size_t> assignments(N);

        for (int iter = 0; iter < KMEANS_ITERS; ++iter) {
            // Assignment: cada ponto ao centroide mais próximo via simd_dist.
            for (size_t i = 0; i < N; ++i) {
                int32_t best_dist = INT32_MAX;
                size_t  best_c   = 0;
                for (size_t c = 0; c < K; ++c) {
                    int32_t d = simd_dist(vecs[i].e, centroids_[c].elements);
                    if (d < best_dist) { best_dist = d; best_c = c; }
                }
                assignments[i] = best_c;
            }

            // Update: acumula em int64 para evitar overflow (N * SCALE^2 > INT32_MAX).
            struct Acc { int64_t sum[16]; size_t count; };
            std::vector<Acc> accum(K, {{}, 0});

            for (size_t i = 0; i < N; ++i) {
                size_t c = assignments[i];
                accum[c].count++;
                for (int j = 0; j < 16; ++j)
                    accum[c].sum[j] += vecs[i].e[j];
            }

            for (size_t c = 0; c < K; ++c) {
                if (accum[c].count == 0) continue;
                for (int j = 0; j < 16; ++j)
                    centroids_[c].elements[j] = static_cast<int16_t>(
                        accum[c].sum[j] / static_cast<int64_t>(accum[c].count));
            }
        }

        total_vectors_ = 0;
        for (auto& b : buckets_) b.clear();

        for (size_t i = 0; i < N; ++i) {
            PaddedVector pv;
            std::copy(vecs[i].e, vecs[i].e + 16, pv.elements);
            pv.norm = compute_norm(pv.elements);
            pv.is_fraud = data[i].second;
            buckets_[assignments[i]].push_back(pv);
            total_vectors_++;
        }

        // Ordena cada bucket por norm (ascendente) para early termination via bail_norm.
        for (auto& b : buckets_) {
            std::sort(b.begin(), b.end(),
                [](const PaddedVector& a, const PaddedVector& b_) { return a.norm < b_.norm; });
            b.shrink_to_fit();
        }

        logging::Logger::info("Index ready: vectors=" + std::to_string(total_vectors_) +
                              " clusters=" + std::to_string(K));
    }

    // Busca KNN em 2 fases com norm pruning e prefetch:
    //   1. partial_sort sobre near_clusters para PROBE_CLUSTERS centroides.
    //   2. Scan nos buckets selecionados com:
    //      a) Lower-bound via norma (desigualdade triangular) → pula sem SIMD
    //      b) Early termination via bail_norm (buckets sorted) → para o bucket
    //      c) Prefetch de cache lines PREFETCH_AHEAD vetores à frente
    //      d) Top-k com insertion sort
    std::vector<application::ports::out::SearchResult>
    search(const domain::Vector14& query_vector, int k) override {
        if (total_vectors_ == 0) return {};

        // Quantiza query com arredondamento
        alignas(32) int16_t q[16];
        for (int i = 0; i < 14; ++i) q[i] = quantize(query_vector[i]);
        q[14] = 0; q[15] = 0;

        // Computa norma da query para pruning
        uint32_t q_norm_sq = 0;
        for (int i = 0; i < 14; ++i) {
            int32_t v = static_cast<int32_t>(q[i]);
            q_norm_sq += static_cast<uint32_t>(v * v);
        }
        uint32_t q_norm;
        {
            double s = std::sqrt(static_cast<double>(q_norm_sq));
            auto up = static_cast<uint32_t>(s);
            if (static_cast<double>(up) < s) ++up;
            q_norm = up;
        }

        // Fase 1: encontra os PROBE_CLUSTERS centroides mais próximos
        struct ClusterDist { int32_t dist; uint16_t id; };
        ClusterDist near_clusters[NUM_CLUSTERS];

        const size_t nc = centroids_.size();
        for (size_t c = 0; c < nc; ++c)
            near_clusters[c] = { simd_dist(q, centroids_[c].elements), static_cast<uint16_t>(c) };

        std::partial_sort(near_clusters, near_clusters + PROBE_CLUSTERS,
                          near_clusters + nc,
                          [](const ClusterDist& a, const ClusterDist& b){ return a.dist < b.dist; });

        // Fase 2: scan com norm pruning + prefetch
        struct Candidate { int32_t dist; bool is_fraud; };
        Candidate top_k[5];
        int count = 0;
        int32_t threshold = INT32_MAX;
        uint32_t bail_norm = UINT32_MAX;

        for (size_t p = 0; p < PROBE_CLUSTERS && p < nc; ++p) {
            const auto& bucket = buckets_[near_clusters[p].id];
            const size_t bsize = bucket.size();
            if (bsize == 0) continue;

            const PaddedVector* bdata = bucket.data();

            for (size_t vi = 0; vi < bsize; ++vi) {
                // Early termination: bucket sorted by norm; se norm >= bail, todo o resto pula
                if (static_cast<uint32_t>(bdata[vi].norm) >= bail_norm) break;

                // Prefetch: traz cache line do vetor PREFETCH_AHEAD posições à frente
#if defined(__x86_64__) || defined(_M_X64)
                if (vi + PREFETCH_AHEAD < bsize) {
                    _mm_prefetch(reinterpret_cast<const char*>(&bdata[vi + PREFETCH_AHEAD]),
                                 _MM_HINT_T0);
                }
#elif defined(__aarch64__) || defined(_M_ARM64)
                if (vi + PREFETCH_AHEAD < bsize) {
                    __builtin_prefetch(&bdata[vi + PREFETCH_AHEAD], 0, 3);
                }
#endif

                // Lower-bound pruning via desigualdade triangular:
                // ||q - r|| >= |norm(q) - norm(r)|, portanto ||q-r||^2 >= (norm_q - norm_r)^2
                {
                    int32_t ndiff = static_cast<int32_t>(q_norm) - static_cast<int32_t>(bdata[vi].norm);
                    int32_t lb = ndiff * ndiff;
                    if (lb >= threshold) continue;
                }

                // Distância SIMD completa (só se o lower-bound passou)
                int32_t distance = simd_dist_pv(bdata[vi].elements, q);
                if (distance >= threshold) continue;

                if (count < k) {
                    top_k[count++] = {distance, bdata[vi].is_fraud};
                    if (count == k) {
                        std::sort(top_k, top_k + k,
                            [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                        threshold = top_k[k-1].dist;
                        bail_norm = compute_bail_norm(q_norm, threshold);
                    }
                } else {
                    top_k[k-1] = {distance, bdata[vi].is_fraud};
                    for (int l = k-1; l > 0 && top_k[l].dist < top_k[l-1].dist; --l)
                        std::swap(top_k[l], top_k[l-1]);
                    threshold = top_k[k-1].dist;
                    bail_norm = compute_bail_norm(q_norm, threshold);
                }
            }
        }

        std::vector<application::ports::out::SearchResult> results;
        results.reserve(count);
        for (int i = 0; i < count; ++i)
            results.push_back({top_k[i].is_fraud, static_cast<float>(top_k[i].dist)});
        return results;
    }

    size_t get_total_vectors() const { return total_vectors_; }

    void log_memory_stats() const {
        auto fmt_mb = [](size_t bytes) -> std::string {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
            return std::string(buf);
        };

        const size_t centroids_bytes = centroids_.size() * sizeof(Centroid);
        const size_t buckets_meta    = buckets_.size()   * sizeof(std::vector<PaddedVector>);
        const size_t buckets_data    = total_vectors_    * sizeof(PaddedVector);
        const size_t total_index     = centroids_bytes + buckets_meta + buckets_data;

        logging::Logger::info("[mem] sizeof(Centroid)=" + std::to_string(sizeof(Centroid)) +
                              "B  sizeof(PaddedVector)=" + std::to_string(sizeof(PaddedVector)) + "B");
        logging::Logger::info("[mem] clusters=" + std::to_string(centroids_.size()) +
                              "  vectors=" + std::to_string(total_vectors_));
        logging::Logger::info("[mem] centroids       =" + fmt_mb(centroids_bytes));
        logging::Logger::info("[mem] buckets_metadata=" + fmt_mb(buckets_meta) +
                              "  (" + std::to_string(buckets_.size()) + " x " +
                              std::to_string(sizeof(std::vector<PaddedVector>)) + "B header)");
        logging::Logger::info("[mem] buckets_data    =" + fmt_mb(buckets_data));
        logging::Logger::info("[mem] total_index     =" + fmt_mb(total_index));
    }

    // Aplica madvise(MADV_HUGEPAGE) nos dados dos buckets para reduzir TLB misses.
    // Deve ser chamado após load_binary() ou train_and_build().
    void apply_hugepages() {
#ifdef __linux__
        for (auto& b : buckets_) {
            if (!b.empty()) {
                void* ptr = static_cast<void*>(b.data());
                size_t len = b.size() * sizeof(PaddedVector);
                // Alinha ao page boundary (4KB)
                uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
                uintptr_t aligned = addr & ~(4095ULL);
                size_t adj_len = len + (addr - aligned);
                ::madvise(reinterpret_cast<void*>(aligned), adj_len, MADV_HUGEPAGE);
                ::madvise(reinterpret_cast<void*>(aligned), adj_len, MADV_WILLNEED);
            }
        }
        logging::Logger::info("[mem] madvise(MADV_HUGEPAGE) applied to bucket data");
#endif
    }

    void save_binary(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            logging::Logger::error("Failed to open for write: " + path);
            return;
        }

        // Header: versão do formato (para compatibilidade futura)
        uint32_t version = 2;  // v2: inclui campo norm
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));

        size_t nc = centroids_.size();
        out.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
        out.write(reinterpret_cast<const char*>(centroids_.data()), nc * sizeof(Centroid));

        for (const auto& b : buckets_) {
            size_t bs = b.size();
            out.write(reinterpret_cast<const char*>(&bs), sizeof(bs));
            if (bs > 0)
                out.write(reinterpret_cast<const char*>(b.data()), bs * sizeof(PaddedVector));
        }
    }

    bool load_binary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        uint32_t version;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 2) {
            logging::Logger::error("Binary index version mismatch (expected 2, got " +
                                    std::to_string(version) + "). Rebuild with --prepare.");
            return false;
        }

        size_t nc;
        in.read(reinterpret_cast<char*>(&nc), sizeof(nc));
        centroids_.resize(nc);
        in.read(reinterpret_cast<char*>(centroids_.data()), nc * sizeof(Centroid));

        total_vectors_ = 0;
        buckets_.clear();
        buckets_.resize(nc);
        for (size_t i = 0; i < nc; ++i) {
            size_t bs;
            in.read(reinterpret_cast<char*>(&bs), sizeof(bs));
            buckets_[i].resize(bs);
            if (bs > 0)
                in.read(reinterpret_cast<char*>(buckets_[i].data()), bs * sizeof(PaddedVector));
            total_vectors_ += bs;
        }

        bool ok = static_cast<bool>(in);
        if (ok) apply_hugepages();
        return ok;
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
