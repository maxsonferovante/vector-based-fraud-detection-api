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

namespace infrastructure {
namespace adapters {
namespace vector_search {

// IVF (Inverted File Index) com busca SIMD AVX2/NEON.
// Índice construído offline via train_and_build (--prepare);
// search() é read-only e thread-safe após a construção.
class SimdIvfMatcher : public application::ports::out::VectorSearchPort {
public:
    // alignas removido: sizeof = 34 bytes → 3M × 34 = ~97 MB (seguro dentro de 168 MB).
    // alignas(32) causava sizeof = 64 bytes → 3M × 64 = 183 MB → OOM no Linux cgroup.
    struct PaddedVector {
        int16_t elements[16];
        bool is_fraud;
    };

private:
    static constexpr size_t   NUM_CLUSTERS   = 2048;
    static constexpr size_t   PROBE_CLUSTERS = 16;
    static constexpr float    SCALE          = 4096.0f;
    static constexpr int      KMEANS_ITERS   = 3;

    struct alignas(32) Centroid {
        int16_t elements[16];
    };

    std::vector<Centroid>                   centroids_;
    std::vector<std::vector<PaddedVector>>  buckets_;
    size_t total_vectors_ = 0;

    // simd_dist: distância euclidiana quadrada para ponteiros 32-byte aligned (Centroid).
    // Ponteiros devem estar alinhados a 32 bytes (__restrict__ evita alias check).
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
    // loadu vs load: sem penalidade em AVX2 quando o dado não cruza boundary de cache line.
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
    // Fluxo: random seed → KMEANS_ITERS x (assignment SIMD + update int64) → fill buckets.
    void train_and_build(const std::vector<std::pair<std::array<float, 14>, bool>>& data) {
        if (data.empty()) return;

        const size_t N = data.size();
        const size_t K = std::min(NUM_CLUSTERS, N);

        logging::Logger::info("Building IVF index: k-means iters=" +
                              std::to_string(KMEANS_ITERS) + " clusters=" +
                              std::to_string(K));

        // Converte float→int16 uma vez; alinhado a 32 bytes para loads SIMD sem penalidade.
        struct alignas(32) AlignedVec { int16_t e[16]; };
        std::vector<AlignedVec> vecs(N);
        for (size_t i = 0; i < N; ++i) {
            const auto& v = data[i].first;
            for (int j = 0; j < 14; ++j)
                vecs[i].e[j] = static_cast<int16_t>(v[j] * SCALE);
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
            pv.is_fraud = data[i].second;
            buckets_[assignments[i]].push_back(pv);
            total_vectors_++;
        }

        logging::Logger::info("Index ready: vectors=" + std::to_string(total_vectors_) +
                              " clusters=" + std::to_string(K));
    }

    // Busca KNN em 2 fases:
    //   1. partial_sort sobre near_clusters (stack array, ~16 KB) para PROBE_CLUSTERS centroides.
    //   2. Scan exaustivo SIMD nos buckets selecionados, mantendo top-k com insertion sort.
    std::vector<application::ports::out::SearchResult>
    search(const domain::Vector14& query_vector, int k) override {
        if (total_vectors_ == 0) return {};

        alignas(32) int16_t q[16];
        for (int i = 0; i < 14; ++i) q[i] = static_cast<int16_t>(query_vector[i] * SCALE);
        q[14] = 0; q[15] = 0;

        struct ClusterDist { int32_t dist; uint16_t id; };
        // Array no stack: NUM_CLUSTERS * sizeof(ClusterDist) = 2048 * 6 ≈ 12 KB.
        ClusterDist near_clusters[NUM_CLUSTERS];

        const size_t nc = centroids_.size();
        for (size_t c = 0; c < nc; ++c)
            near_clusters[c] = { simd_dist(q, centroids_[c].elements), static_cast<uint16_t>(c) };

        std::partial_sort(near_clusters, near_clusters + PROBE_CLUSTERS,
                          near_clusters + nc,
                          [](const ClusterDist& a, const ClusterDist& b){ return a.dist < b.dist; });

        struct Candidate { int32_t dist; bool is_fraud; };
        Candidate top_k[5];
        int count = 0;
        int32_t threshold = INT32_MAX;

        for (size_t p = 0; p < PROBE_CLUSTERS && p < nc; ++p) {
            for (const auto& pv : buckets_[near_clusters[p].id]) {
                int32_t distance = simd_dist_pv(q, pv.elements); // loadu: PaddedVector sem alignas
                if (distance >= threshold) continue;

                if (count < k) {
                    top_k[count++] = {distance, pv.is_fraud};
                    if (count == k) {
                        std::sort(top_k, top_k + k,
                            [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                        threshold = top_k[k-1].dist;
                    }
                } else {
                    top_k[k-1] = {distance, pv.is_fraud};
                    for (int l = k-1; l > 0 && top_k[l].dist < top_k[l-1].dist; --l)
                        std::swap(top_k[l], top_k[l-1]);
                    threshold = top_k[k-1].dist;
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

    void save_binary(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            logging::Logger::error("Failed to open for write: " + path);
            return;
        }
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
        return static_cast<bool>(in);
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
