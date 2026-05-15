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
private:
    static constexpr size_t   NUM_CLUSTERS   = 1023;
    static constexpr size_t   PROBE_CLUSTERS = 32;  // Increased from 16 for better recall
    static constexpr float    SCALE          = 4096.0f;
    static constexpr int      KMEANS_ITERS   = 5;
    static constexpr size_t   PREFETCH_AHEAD = 32;

    struct alignas(32) Centroid {
        int16_t elements[16];
    };

    // SoA (Struct of Arrays) format for better cache locality and SIMD alignment
    struct Bucket {
        std::vector<Centroid> elements; // Each Centroid is 16 int16_t, 32-byte aligned
        std::vector<uint16_t> norms;
        std::vector<uint8_t>  labels;
    };

    std::vector<Centroid> centroids_;
    std::vector<Bucket>   buckets_;
    size_t total_vectors_ = 0;

    static inline int16_t quantize(float v) noexcept {
        float scaled = v * SCALE;
        return (v >= 0.0f)
            ? static_cast<int16_t>(scaled + 0.5f)
            : static_cast<int16_t>(scaled - 0.5f);
    }

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

    static inline uint32_t compute_bail_norm(uint32_t q_norm, int32_t threshold) noexcept {
        if (threshold == INT32_MAX) return UINT32_MAX;
        double s = std::sqrt(static_cast<double>(threshold));
        auto up = static_cast<uint32_t>(s);
        if (static_cast<double>(up) < s) ++up;
        uint64_t b = static_cast<uint64_t>(q_norm) + up + 1u;
        return (b > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(b);
    }

    static inline int32_t simd_dist(const int16_t* __restrict__ a,
                                    const int16_t* __restrict__ b) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        __m256i va  = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
        __m256i vb  = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
        __m256i d   = _mm256_sub_epi16(va, vb);
        __m256i sq  = _mm256_madd_epi16(d, d);
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

public:
    SimdIvfMatcher() {
        buckets_.resize(NUM_CLUSTERS);
        for (auto& b : buckets_) {
            b.elements.reserve(2000);
            b.norms.reserve(2000);
            b.labels.reserve(2000);
        }
    }

    void train_and_build(const std::vector<std::pair<std::array<float, 14>, bool>>& data) {
        if (data.empty()) return;

        const size_t N = data.size();
        const size_t K = std::min(NUM_CLUSTERS, N);

        logging::Logger::info("Building IVF index: k-means iters=" +
                              std::to_string(KMEANS_ITERS) + " clusters=" +
                              std::to_string(K));

        struct alignas(32) AlignedVec { int16_t e[16]; };
        std::vector<AlignedVec> vecs(N);
        for (size_t i = 0; i < N; ++i) {
            const auto& v = data[i].first;
            for (int j = 0; j < 14; ++j)
                vecs[i].e[j] = quantize(v[j]);
            vecs[i].e[14] = 0; vecs[i].e[15] = 0;
        }

        std::mt19937 rng(42);
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        centroids_.resize(K);
        for (size_t i = 0; i < K; ++i)
            std::copy(vecs[indices[i]].e, vecs[indices[i]].e + 16, centroids_[i].elements);

        std::vector<size_t> assignments(N);

        for (int iter = 0; iter < KMEANS_ITERS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                int32_t best_dist = INT32_MAX;
                size_t  best_c   = 0;
                for (size_t c = 0; c < K; ++c) {
                    int32_t d = simd_dist(vecs[i].e, centroids_[c].elements);
                    if (d < best_dist) { best_dist = d; best_c = c; }
                }
                assignments[i] = best_c;
            }

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
        for (auto& b : buckets_) {
            b.elements.clear();
            b.norms.clear();
            b.labels.clear();
        }

        // Temporal struct for sorting
        struct SortingItem {
            Centroid elements;
            uint16_t norm;
            bool is_fraud;
        };
        std::vector<std::vector<SortingItem>> temp_buckets(K);

        for (size_t i = 0; i < N; ++i) {
            SortingItem item;
            std::copy(vecs[i].e, vecs[i].e + 16, item.elements.elements);
            item.norm = compute_norm(item.elements.elements);
            item.is_fraud = data[i].second;
            temp_buckets[assignments[i]].push_back(item);
            total_vectors_++;
        }

        for (size_t i = 0; i < K; ++i) {
            std::sort(temp_buckets[i].begin(), temp_buckets[i].end(),
                [](const SortingItem& a, const SortingItem& b) { return a.norm < b.norm; });
            
            for (const auto& item : temp_buckets[i]) {
                buckets_[i].elements.push_back(item.elements);
                buckets_[i].norms.push_back(item.norm);
                buckets_[i].labels.push_back(item.is_fraud ? 1 : 0);
            }
            buckets_[i].elements.shrink_to_fit();
            buckets_[i].norms.shrink_to_fit();
            buckets_[i].labels.shrink_to_fit();
        }

        logging::Logger::info("Index ready: vectors=" + std::to_string(total_vectors_) +
                              " clusters=" + std::to_string(K));
    }

    application::ports::out::SearchResultList<5>
    search(const domain::Vector14& query_vector, int k) override {
        application::ports::out::SearchResultList<5> results;
        if (total_vectors_ == 0) return results;

        alignas(32) int16_t q[16];
        for (int i = 0; i < 14; ++i) q[i] = quantize(query_vector[i]);
        q[14] = 0; q[15] = 0;

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

        struct ClusterDist { int32_t dist; uint16_t id; };
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
        uint32_t bail_norm = UINT32_MAX;

        for (size_t p = 0; p < PROBE_CLUSTERS && p < nc; ++p) {
            const auto& bucket = buckets_[near_clusters[p].id];
            const size_t bsize = bucket.elements.size();
            if (bsize == 0) continue;

            const Centroid* b_elements = bucket.elements.data();
            const uint16_t* b_norms = bucket.norms.data();
            const uint8_t*  b_labels = bucket.labels.data();

            for (size_t vi = 0; vi < bsize; ++vi) {
                if (static_cast<uint32_t>(b_norms[vi]) >= bail_norm) break;

#if defined(__x86_64__) || defined(_M_X64)
                if (vi + PREFETCH_AHEAD < bsize) {
                    _mm_prefetch(reinterpret_cast<const char*>(&b_elements[vi + PREFETCH_AHEAD]),
                                 _MM_HINT_T0);
                }
#elif defined(__aarch64__) || defined(_M_ARM64)
                if (vi + PREFETCH_AHEAD < bsize) {
                    __builtin_prefetch(&b_elements[vi + PREFETCH_AHEAD], 0, 3);
                }
#endif

                {
                    int32_t ndiff = static_cast<int32_t>(q_norm) - static_cast<int32_t>(b_norms[vi]);
                    int32_t lb = ndiff * ndiff;
                    if (lb >= threshold) continue;
                }

                // Now elements are aligned, we can use simd_dist directly (aligned load)
                int32_t distance = simd_dist(b_elements[vi].elements, q);
                if (distance >= threshold) continue;

                if (count < k) {
                    top_k[count++] = {distance, b_labels[vi] != 0};
                    if (count == k) {
                        std::sort(top_k, top_k + k,
                            [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                        threshold = top_k[k-1].dist;
                        bail_norm = compute_bail_norm(q_norm, threshold);
                    }
                } else {
                    top_k[k-1] = {distance, b_labels[vi] != 0};
                    for (int l = k-1; l > 0 && top_k[l].dist < top_k[l-1].dist; --l)
                        std::swap(top_k[l], top_k[l-1]);
                    threshold = top_k[k-1].dist;
                    bail_norm = compute_bail_norm(q_norm, threshold);
                }
            }
        }

        for (int i = 0; i < count; ++i) {
            results.items[i] = {top_k[i].is_fraud, static_cast<float>(top_k[i].dist)};
        }
        results.count = count;
        return results;
    }

    size_t get_total_vectors() const { return total_vectors_; }

    void save_binary(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            logging::Logger::error("Failed to open for write: " + path);
            return;
        }

        uint32_t version = 3;  // v3: SoA format
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));

        size_t nc = centroids_.size();
        out.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
        out.write(reinterpret_cast<const char*>(centroids_.data()), nc * sizeof(Centroid));

        for (const auto& b : buckets_) {
            size_t bs = b.elements.size();
            out.write(reinterpret_cast<const char*>(&bs), sizeof(bs));
            if (bs > 0) {
                out.write(reinterpret_cast<const char*>(b.elements.data()), bs * sizeof(Centroid));
                out.write(reinterpret_cast<const char*>(b.norms.data()), bs * sizeof(uint16_t));
                out.write(reinterpret_cast<const char*>(b.labels.data()), bs * sizeof(uint8_t));
            }
        }
    }

    bool load_binary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        uint32_t version;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 3) {
            logging::Logger::error("Binary index version mismatch (expected 3, got " +
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
            if (bs > 0) {
                buckets_[i].elements.resize(bs);
                buckets_[i].norms.resize(bs);
                buckets_[i].labels.resize(bs);
                in.read(reinterpret_cast<char*>(buckets_[i].elements.data()), bs * sizeof(Centroid));
                in.read(reinterpret_cast<char*>(buckets_[i].norms.data()), bs * sizeof(uint16_t));
                in.read(reinterpret_cast<char*>(buckets_[i].labels.data()), bs * sizeof(uint8_t));
            }
            total_vectors_ += bs;
        }
        return static_cast<bool>(in);;
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
