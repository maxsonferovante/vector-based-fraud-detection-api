#ifndef SIMD_IVF_MATCHER_HPP
#define SIMD_IVF_MATCHER_HPP

#include "../../../application/ports/out/vector_search_port.hpp"
#include "../../logging/logger.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <fstream>
#include <random>
#include <numeric>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

namespace infrastructure {
namespace adapters {
namespace vector_search {

class SimdIvfMatcher : public application::ports::out::VectorSearchPort {
public:
    struct alignas(32) PaddedVector {
        int16_t elements[16];
        bool is_fraud;
    };

private:
    static constexpr size_t NUM_CLUSTERS = 2048; // Aumentado para 2k
    static constexpr size_t PROBE_CLUSTERS = 16; // Aumentado para 16 probes para precisão Top-1
    static constexpr float SCALE = 4096.0f;

    struct alignas(32) Centroid {
        int16_t elements[16];
    };

    std::vector<Centroid> centroids_;
    std::vector<std::vector<PaddedVector>> buckets_;
    size_t total_vectors_ = 0;

public:
    SimdIvfMatcher() {
        buckets_.resize(NUM_CLUSTERS);
        for(auto& b : buckets_) b.reserve(2000); 
    }

    void train_and_build(const std::vector<std::pair<std::array<float, 14>, bool>>& data) {
        if (data.empty()) return;
        
        logging::Logger::info("Building Elite-IVF index with " + std::to_string(NUM_CLUSTERS) + " clusters...");

        std::mt19937 rng(42);
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        centroids_.clear();
        for (size_t i = 0; i < NUM_CLUSTERS && i < data.size(); ++i) {
            Centroid c;
            const auto& v = data[indices[i]].first;
            for (int j = 0; j < 14; ++j) c.elements[j] = static_cast<int16_t>(v[j] * SCALE);
            c.elements[14] = 0; c.elements[15] = 0;
            centroids_.push_back(c);
        }

        for (const auto& item : data) {
            const auto& v = item.first;
            bool is_fraud = item.second;
            
            int16_t q[16];
            for (int j = 0; j < 14; ++j) q[j] = static_cast<int16_t>(v[j] * SCALE);
            q[14] = 0; q[15] = 0;

            int best_cluster = 0;
            int32_t min_dist = 2147483647;

            for (size_t c = 0; c < centroids_.size(); ++c) {
                int32_t d = 0;
                for (int j = 0; j < 14; ++j) {
                    int32_t diff = (int32_t)q[j] - (int32_t)centroids_[c].elements[j];
                    d += diff * diff;
                }
                if (d < min_dist) { min_dist = d; best_cluster = static_cast<int>(c); }
            }

            PaddedVector pv;
            std::copy(q, q + 16, pv.elements);
            pv.is_fraud = is_fraud;
            buckets_[best_cluster].push_back(pv);
            total_vectors_++;
        }
        logging::Logger::info("IVF build complete. Total vectors: " + std::to_string(total_vectors_));
    }

    std::vector<application::ports::out::SearchResult> search(const domain::Vector14& query_vector, int k) override {
        if (total_vectors_ == 0) return {};

        alignas(32) int16_t q[16];
        for (int i = 0; i < 14; ++i) q[i] = static_cast<int16_t>(query_vector[i] * SCALE);
        q[14] = 0; q[15] = 0;

#if defined(__x86_64__) || defined(_M_X64)
        __m256i v_query = _mm256_load_si256((__m256i*)q);
#endif

        struct ClusterDist { int32_t dist; size_t id; };
        std::vector<ClusterDist> near_clusters(centroids_.size());
        
        // Fase 1: Busca de Centroids Otimizada (também via SIMD se possível)
        for (size_t c = 0; c < centroids_.size(); ++c) {
            int32_t d;
#if defined(__x86_64__) || defined(_M_X64)
            __m256i v_c = _mm256_load_si256((__m256i*)&centroids_[c]);
            __m256i diff = _mm256_sub_epi16(v_query, v_c);
            __m256i squared = _mm256_madd_epi16(diff, diff);
            __m128i low = _mm256_castsi256_si128(squared);
            __m128i high = _mm256_extracti128_si256(squared, 1);
            __m128i sum128 = _mm_add_epi32(low, high);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
            d = _mm_cvtsi128_si32(sum128);
#else
            d = 0;
            for (int j = 0; j < 14; ++j) {
                int32_t diff = (int32_t)q[j] - (int32_t)centroids_[c].elements[j];
                d += diff * diff;
            }
#endif
            near_clusters[c] = {d, c};
        }
        
        std::partial_sort(near_clusters.begin(), near_clusters.begin() + PROBE_CLUSTERS, near_clusters.end(), 
            [](const ClusterDist& a, const ClusterDist& b){ return a.dist < b.dist; });

        // Fase 2: Busca SIMD nos Buckets selecionados
        struct Candidate { int32_t dist; bool is_fraud; };
        Candidate top_k[5];
        int count = 0;
        int32_t threshold = 2147483647;

        for (size_t p = 0; p < PROBE_CLUSTERS; ++p) {
            const auto& bucket = buckets_[near_clusters[p].id];
            for (const auto& pv : bucket) {
                int32_t distance;
#if defined(__x86_64__) || defined(_M_X64)
                __m256i v_ref = _mm256_load_si256((__m256i*)pv.elements);
                __m256i diff = _mm256_sub_epi16(v_query, v_ref);
                __m256i squared = _mm256_madd_epi16(diff, diff);
                __m128i low = _mm256_castsi256_si128(squared);
                __m128i high = _mm256_extracti128_si256(squared, 1);
                __m128i sum128 = _mm_add_epi32(low, high);
                sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
                sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
                distance = _mm_cvtsi128_si32(sum128);
#else
                distance = 0;
                for (int j = 0; j < 14; ++j) {
                    int32_t d = (int32_t)q[j] - (int32_t)pv.elements[j];
                    distance += d * d;
                }
#endif
                if (distance < threshold) {
                    if (count < k) {
                        top_k[count++] = {distance, pv.is_fraud};
                        if (count == k) {
                            std::sort(top_k, top_k + k, [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                            threshold = top_k[k-1].dist;
                        }
                    } else {
                        top_k[k-1] = {distance, pv.is_fraud};
                        for (int l = k - 1; l > 0 && top_k[l].dist < top_k[l-1].dist; --l) std::swap(top_k[l], top_k[l-1]);
                        threshold = top_k[k-1].dist;
                    }
                }
            }
        }

        std::vector<application::ports::out::SearchResult> results;
        for (int i = 0; i < count; ++i) results.push_back({top_k[i].is_fraud, (float)top_k[i].dist});
        return results;
    }

    size_t get_total_vectors() const { return total_vectors_; }
    
    void save_binary(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        size_t nc = centroids_.size();
        out.write((char*)&nc, sizeof(nc));
        out.write((char*)centroids_.data(), nc * sizeof(Centroid));
        for(const auto& b : buckets_) {
            size_t bs = b.size();
            out.write((char*)&bs, sizeof(bs));
            out.write((char*)b.data(), bs * sizeof(PaddedVector));
        }
    }

    bool load_binary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        size_t nc;
        in.read((char*)&nc, sizeof(nc));
        centroids_.resize(nc);
        in.read((char*)centroids_.data(), nc * sizeof(Centroid));
        total_vectors_ = 0;
        buckets_.clear();
        buckets_.resize(nc);
        for(size_t i = 0; i < nc; ++i) {
            size_t bs;
            in.read((char*)&bs, sizeof(bs));
            buckets_[i].resize(bs);
            in.read((char*)buckets_[i].data(), bs * sizeof(PaddedVector));
            total_vectors_ += bs;
        }
        return true;
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
