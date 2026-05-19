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
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
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
    static constexpr size_t   PROBE_CLUSTERS = 64;
    static constexpr int      KMEANS_ITERS   = 5;
    static constexpr size_t   PREFETCH_AHEAD = 32;

    struct alignas(64) Centroid {
        float elements[16];
    };

    // BucketView represents a window into the vector data.
    // In build mode, it points to vectors. In mmap mode, it points to the mapped file.
    struct BucketView {
        size_t size = 0;
        const Centroid* elements = nullptr;
        const float* norms = nullptr;
        const uint8_t*  labels = nullptr;
    };

    // Storage for build mode (prepare)
    struct Bucket {
        std::vector<Centroid> elements;
        std::vector<float> norms;
        std::vector<uint8_t>  labels;
    };

    std::vector<Centroid> centroids_storage_;
    std::vector<Bucket>   buckets_storage_;

    // Views used for search
    const Centroid* centroids_ = nullptr;
    std::vector<BucketView> buckets_;
    size_t total_vectors_ = 0;

    // mmap resources
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;

    void cleanup_mmap() {
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
            munmap(mmap_ptr_, mmap_size_);
        }
        mmap_ptr_ = nullptr;
        mmap_size_ = 0;
        centroids_ = nullptr;
        buckets_.clear();
        total_vectors_ = 0;
    }

    static inline float compute_norm(const float* v) noexcept {
        float sum = 0.0f;
        for (int i = 0; i < 14; ++i) {
            sum += v[i] * v[i];
        }
        return std::sqrt(sum);
    }

    static inline float compute_bail_norm(float q_norm, float threshold) noexcept {
        if (threshold == std::numeric_limits<float>::max()) return std::numeric_limits<float>::max();
        return q_norm + std::sqrt(threshold);
    }

    static inline float simd_dist(const float* __restrict__ a,
                                  const float* __restrict__ b) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        __m256 va0 = _mm256_load_ps(a);
        __m256 vb0 = _mm256_load_ps(b);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
        __m256 sq0 = _mm256_mul_ps(d0, d0);

        __m256 va1 = _mm256_load_ps(a + 8);
        __m256 vb1 = _mm256_load_ps(b + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
        __m256 sq1 = _mm256_mul_ps(d1, d1);

        __m256 sum256 = _mm256_add_ps(sq0, sq1);
        
        // Horizontal add
        __m128 vlow = _mm256_castps256_ps128(sum256);
        __m128 vhigh = _mm256_extractf128_ps(sum256, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_add_ps(vlow, _mm_movehl_ps(vlow, vlow));
        vlow = _mm_add_ss(vlow, _mm_shuffle_ps(vlow, vlow, 1));
        return _mm_cvtss_f32(vlow);
#elif defined(__aarch64__) || defined(_M_ARM64)
        float32x4_t va0 = vld1q_f32(a);
        float32x4_t va1 = vld1q_f32(a + 4);
        float32x4_t va2 = vld1q_f32(a + 8);
        float32x4_t va3 = vld1q_f32(a + 12);

        float32x4_t vb0 = vld1q_f32(b);
        float32x4_t vb1 = vld1q_f32(b + 4);
        float32x4_t vb2 = vld1q_f32(b + 8);
        float32x4_t vb3 = vld1q_f32(b + 12);

        float32x4_t d0 = vsubq_f32(va0, vb0);
        float32x4_t d1 = vsubq_f32(va1, vb1);
        float32x4_t d2 = vsubq_f32(va2, vb2);
        float32x4_t d3 = vsubq_f32(va3, vb3);

        float32x4_t s0 = vmulq_f32(d0, d0);
        s0 = vmlaq_f32(s0, d1, d1);
        s0 = vmlaq_f32(s0, d2, d2);
        s0 = vmlaq_f32(s0, d3, d3);

        return vaddvq_f32(s0);
#else
        float dist = 0.0f;
        for (int j = 0; j < 14; ++j) {
            float d = a[j] - b[j];
            dist += d * d;
        }
        return dist;
#endif
    }

public:
    SimdIvfMatcher() {
        buckets_storage_.resize(NUM_CLUSTERS);
        for (auto& b : buckets_storage_) {
            b.elements.reserve(2000);
            b.norms.reserve(2000);
            b.labels.reserve(2000);
        }
        buckets_.resize(NUM_CLUSTERS);
    }

    ~SimdIvfMatcher() {
        cleanup_mmap();
    }

    void train_and_build(const std::vector<std::pair<std::array<float, 14>, bool>>& data) {
        if (data.empty()) return;

        const size_t N = data.size();
        const size_t K = std::min(NUM_CLUSTERS, N);

        logging::Logger::info("Building IVF index: k-means iters=" +
                              std::to_string(KMEANS_ITERS) + " clusters=" +
                              std::to_string(K));

        struct alignas(64) AlignedVec { float e[16]; };
        std::vector<AlignedVec> vecs(N);
        for (size_t i = 0; i < N; ++i) {
            const auto& v = data[i].first;
            for (int j = 0; j < 14; ++j)
                vecs[i].e[j] = v[j];
            vecs[i].e[14] = 0.0f; vecs[i].e[15] = 0.0f;
        }

        std::mt19937 rng(42);
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        centroids_storage_.resize(K);
        for (size_t i = 0; i < K; ++i)
            std::copy(vecs[indices[i]].e, vecs[indices[i]].e + 16, centroids_storage_[i].elements);

        std::vector<size_t> assignments(N);

        for (int iter = 0; iter < KMEANS_ITERS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                float best_dist = std::numeric_limits<float>::max();
                size_t  best_c   = 0;
                for (size_t c = 0; c < K; ++c) {
                    float d = simd_dist(vecs[i].e, centroids_storage_[c].elements);
                    if (d < best_dist) { best_dist = d; best_c = c; }
                }
                assignments[i] = best_c;
            }

            struct Acc { double sum[16]; size_t count; };
            std::vector<Acc> accum(K);
            for (size_t c = 0; c < K; ++c) {
                for (int j = 0; j < 16; ++j) accum[c].sum[j] = 0.0;
                accum[c].count = 0;
            }

            for (size_t i = 0; i < N; ++i) {
                size_t c = assignments[i];
                accum[c].count++;
                for (int j = 0; j < 16; ++j)
                    accum[c].sum[j] += static_cast<double>(vecs[i].e[j]);
            }

            for (size_t c = 0; c < K; ++c) {
                if (accum[c].count == 0) continue;
                for (int j = 0; j < 16; ++j)
                    centroids_storage_[c].elements[j] = static_cast<float>(
                        accum[c].sum[j] / static_cast<double>(accum[c].count));
            }
        }

        total_vectors_ = 0;
        for (auto& b : buckets_storage_) {
            b.elements.clear();
            b.norms.clear();
            b.labels.clear();
        }

        // Temporal struct for sorting
        struct SortingItem {
            Centroid elements;
            float norm;
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

        centroids_ = centroids_storage_.data();
        buckets_.resize(K);

        for (size_t i = 0; i < K; ++i) {
            std::sort(temp_buckets[i].begin(), temp_buckets[i].end(),
                [](const SortingItem& a, const SortingItem& b) { return a.norm < b.norm; });
            
            for (const auto& item : temp_buckets[i]) {
                buckets_storage_[i].elements.push_back(item.elements);
                buckets_storage_[i].norms.push_back(item.norm);
                buckets_storage_[i].labels.push_back(item.is_fraud ? 1 : 0);
            }
            buckets_storage_[i].elements.shrink_to_fit();
            buckets_storage_[i].norms.shrink_to_fit();
            buckets_storage_[i].labels.shrink_to_fit();

            // Setup view
            buckets_[i].size = buckets_storage_[i].elements.size();
            buckets_[i].elements = buckets_storage_[i].elements.data();
            buckets_[i].norms = buckets_storage_[i].norms.data();
            buckets_[i].labels = buckets_storage_[i].labels.data();
        }

        logging::Logger::info("Index ready: vectors=" + std::to_string(total_vectors_) +
                              " clusters=" + std::to_string(K));
    }

    application::ports::out::SearchResultList<5>
    search(const domain::Vector14& query_vector, int k) override {
        application::ports::out::SearchResultList<5> results;
        if (total_vectors_ == 0) return results;

        alignas(64) float q[16] = {0.0f};
        for (int i = 0; i < 14; ++i) q[i] = query_vector[i];

        float q_norm = compute_norm(q);

        struct ClusterDist { float dist; uint16_t id; };
        ClusterDist near_clusters[NUM_CLUSTERS];

        const size_t nc = buckets_.size();
        for (size_t c = 0; c < nc; ++c)
            near_clusters[c] = { simd_dist(q, centroids_[c].elements), static_cast<uint16_t>(c) };

        std::partial_sort(near_clusters, near_clusters + PROBE_CLUSTERS,
                          near_clusters + nc,
                          [](const ClusterDist& a, const ClusterDist& b){ return a.dist < b.dist; });

        struct Candidate { float dist; bool is_fraud; };
        Candidate top_k[5];
        int count = 0;
        float threshold = std::numeric_limits<float>::max();
        float bail_norm = std::numeric_limits<float>::max();

        for (size_t p = 0; p < PROBE_CLUSTERS && p < nc; ++p) {
            const auto& bucket = buckets_[near_clusters[p].id];
            const size_t bsize = bucket.size;
            if (bsize == 0) continue;

            const Centroid* b_elements = bucket.elements;
            const float* b_norms = bucket.norms;
            const uint8_t*  b_labels = bucket.labels;

#if defined(__GNUC__) || defined(__clang__)
            b_elements = reinterpret_cast<const Centroid*>(__builtin_assume_aligned(b_elements, 64));
#endif

            for (size_t vi = 0; vi < bsize; ++vi) {
                if (b_norms[vi] >= bail_norm) break;

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
                    float ndiff = q_norm - b_norms[vi];
                    float lb = ndiff * ndiff;
                    if (lb >= threshold) continue;
                }

                // Now elements are aligned, we can use simd_dist directly (aligned load)
                float distance = simd_dist(b_elements[vi].elements, q);
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
            results.items[i] = {top_k[i].is_fraud, top_k[i].dist};
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

        uint32_t version = 5;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        char pad12[12] = {0};
        out.write(pad12, 12);

        size_t nc = buckets_.size();
        out.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
        out.write(reinterpret_cast<const char*>(&total_vectors_), sizeof(total_vectors_));
        // Offset 32.

        // 1. Centroids (starts at 32)
        out.write(reinterpret_cast<const char*>(centroids_), nc * sizeof(Centroid));

        // 2. Bucket sizes
        std::vector<size_t> sizes(nc);
        for (size_t i = 0; i < nc; ++i) sizes[i] = buckets_[i].size;
        out.write(reinterpret_cast<const char*>(sizes.data()), nc * sizeof(size_t));

        // Align to 64 bytes before elements
        size_t current_pos = static_cast<size_t>(out.tellp());
        size_t align_pad = (64 - (current_pos % 64)) % 64;
        if (align_pad > 0) {
            char p[64] = {0};
            out.write(p, align_pad);
        }

        // 3. All Elements
        for (size_t i = 0; i < nc; ++i) {
            if (buckets_[i].size > 0)
                out.write(reinterpret_cast<const char*>(buckets_[i].elements), 
                          buckets_[i].size * sizeof(Centroid));
        }

        // 4. All Norms
        for (size_t i = 0; i < nc; ++i) {
            if (buckets_[i].size > 0)
                out.write(reinterpret_cast<const char*>(buckets_[i].norms), 
                          buckets_[i].size * sizeof(float));
        }

        // 5. All Labels
        for (size_t i = 0; i < nc; ++i) {
            if (buckets_[i].size > 0)
                out.write(reinterpret_cast<const char*>(buckets_[i].labels), 
                          buckets_[i].size * sizeof(uint8_t));
        }
    }

    bool load_binary(const std::string& path) {
        cleanup_mmap();

        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) return false;

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            return false;
        }
        mmap_size_ = sb.st_size;

#ifdef __linux__
        mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
#else
        mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd, 0);
#endif
        close(fd);

        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = nullptr;
            return false;
        }

        const char* base = static_cast<const char*>(mmap_ptr_);
        size_t offset = 0;

        uint32_t version = *reinterpret_cast<const uint32_t*>(base + offset);
        if (version != 5) {
            logging::Logger::error("Binary index version mismatch (expected 5, got " +
                                    std::to_string(version) + "). Rebuild with --prepare.");
            cleanup_mmap();
            return false;
        }
        offset += 16; // skip version and pad

        size_t nc = *reinterpret_cast<const size_t*>(base + offset);
        offset += sizeof(size_t);

        total_vectors_ = *reinterpret_cast<const size_t*>(base + offset);
        offset += sizeof(size_t);

        // 1. Centroids (at 32)
        centroids_ = reinterpret_cast<const Centroid*>(base + offset);
        offset += nc * sizeof(Centroid);

        // 2. Bucket sizes
        const size_t* sizes = reinterpret_cast<const size_t*>(base + offset);
        offset += nc * sizeof(size_t);

        // Align to 64 bytes
        offset = (offset + 63) & ~static_cast<size_t>(63);

        buckets_.resize(nc);

        // 3. All Elements
        const Centroid* all_elements = reinterpret_cast<const Centroid*>(base + offset);
        offset += total_vectors_ * sizeof(Centroid);

        // 4. All Norms
        const float* all_norms = reinterpret_cast<const float*>(base + offset);
        offset += total_vectors_ * sizeof(float);

        // 5. All Labels
        const uint8_t* all_labels = reinterpret_cast<const uint8_t*>(base + offset);

        size_t element_offset = 0;
        size_t norm_offset = 0;
        size_t label_offset = 0;

        for (size_t i = 0; i < nc; ++i) {
            buckets_[i].size = sizes[i];
            if (sizes[i] > 0) {
                buckets_[i].elements = all_elements + element_offset;
                buckets_[i].norms = all_norms + norm_offset;
                buckets_[i].labels = all_labels + label_offset;
                element_offset += sizes[i];
                norm_offset += sizes[i];
                label_offset += sizes[i];
            } else {
                buckets_[i].elements = nullptr;
                buckets_[i].norms = nullptr;
                buckets_[i].labels = nullptr;
            }
        }

        return true;
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
