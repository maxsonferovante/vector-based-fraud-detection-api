#ifndef SIMD_VECTOR_MATCHER_HPP
#define SIMD_VECTOR_MATCHER_HPP

#include "../../../application/ports/out/vector_search_port.hpp"
#include "../../logging/logger.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <fstream>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

namespace infrastructure {
namespace adapters {
namespace vector_search {

/**
 * SIMD-accelerated exhaustive search matcher.
 * Performs linear scan with loop unrolling and architecture-specific intrinsics.
 */
class SimdVectorMatcher : public application::ports::out::VectorSearchPort {
    struct alignas(32) PaddedVector {
        int16_t elements[16]; 
    };

    std::vector<PaddedVector> database_;
    std::vector<int8_t> labels_;
    static constexpr float SCALE = 4096.0f;

public:
    SimdVectorMatcher() {
        database_.reserve(3000000);
        labels_.reserve(3000000);
    }

    void add_vector(const std::array<float, 14>& v, bool is_fraud) {
        PaddedVector pv;
        for (int i = 0; i < 14; ++i) pv.elements[i] = static_cast<int16_t>(v[i] * SCALE);
        pv.elements[14] = 0; pv.elements[15] = 0;
        database_.push_back(pv);
        labels_.push_back(is_fraud ? 1 : 0);
    }

    std::vector<application::ports::out::SearchResult> search(const domain::Vector14& query_vector, int k) override {
        if (database_.empty()) return {};

        alignas(32) int16_t q[16];
        for (int i = 0; i < 14; ++i) q[i] = static_cast<int16_t>(query_vector[i] * SCALE);
        q[14] = 0; q[15] = 0;

        struct Candidate {
            int32_t dist;
            int index;
        };
        Candidate top_k[5]; 
        int count = 0;
        int32_t threshold = 2147483647;

        size_t n = database_.size();
        const PaddedVector* db_ptr = database_.data();

#if defined(__x86_64__) || defined(_M_X64)
        __m256i v_query = _mm256_load_si256((__m256i*)q);
        
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m256i v_ref0 = _mm256_load_si256((__m256i*)&db_ptr[i]);
            __m256i v_ref1 = _mm256_load_si256((__m256i*)&db_ptr[i+1]);
            __m256i v_ref2 = _mm256_load_si256((__m256i*)&db_ptr[i+2]);
            __m256i v_ref3 = _mm256_load_si256((__m256i*)&db_ptr[i+3]);

            auto calc_dist = [&](__m256i v_ref) {
                __m256i diff = _mm256_sub_epi16(v_query, v_ref);
                __m256i squared = _mm256_madd_epi16(diff, diff);
                __m128i low = _mm256_castsi256_si128(squared);
                __m128i high = _mm256_extracti128_si256(squared, 1);
                __m128i sum128 = _mm_add_epi32(low, high);
                sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
                sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
                return _mm_cvtsi128_si32(sum128);
            };

            int32_t d0 = calc_dist(v_ref0);
            int32_t d1 = calc_dist(v_ref1);
            int32_t d2 = calc_dist(v_ref2);
            int32_t d3 = calc_dist(v_ref3);

            auto update_top = [&](int32_t dist, int idx) {
                if (dist < threshold) {
                    if (count < k) {
                        top_k[count++] = {dist, idx};
                        if (count == k) {
                            std::sort(top_k, top_k + k, [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                            threshold = top_k[k-1].dist;
                        }
                    } else {
                        top_k[k-1] = {dist, idx};
                        for (int j = k - 1; j > 0 && top_k[j].dist < top_k[j-1].dist; --j) std::swap(top_k[j], top_k[j-1]);
                        threshold = top_k[k-1].dist;
                    }
                }
            };

            update_top(d0, (int)i);
            update_top(d1, (int)i+1);
            update_top(d2, (int)i+2);
            update_top(d3, (int)i+3);
        }
#elif defined(__aarch64__) || defined(_M_ARM64)
        int16x8_t q_low = vld1q_s16(q);
        int16x8_t q_high = vld1q_s16(q + 8);
        size_t i = 0;
        for (; i < n; ++i) {
            int16x8_t r_low = vld1q_s16(db_ptr[i].elements);
            int16x8_t r_high = vld1q_s16(db_ptr[i].elements + 8);
            int16x8_t diff_low = vsubq_s16(q_low, r_low);
            int16x8_t diff_high = vsubq_s16(q_high, r_high);
            int32x4_t m_low = vmull_s16(vget_low_s16(diff_low), vget_low_s16(diff_low));
            m_low = vmlal_s16(m_low, vget_high_s16(diff_low), vget_high_s16(diff_low));
            int32x4_t m_high = vmull_s16(vget_low_s16(diff_high), vget_low_s16(diff_high));
            m_high = vmlal_s16(m_high, vget_high_s16(diff_high), vget_high_s16(diff_high));
            int32_t distance = vaddvq_s32(vaddq_s32(m_low, m_high));
            
            if (distance < threshold) {
                if (count < k) {
                    top_k[count++] = {distance, (int)i};
                    if (count == k) {
                        std::sort(top_k, top_k + k, [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                        threshold = top_k[k-1].dist;
                    }
                } else {
                    top_k[k-1] = {distance, (int)i};
                    for (int j = k - 1; j > 0 && top_k[j].dist < top_k[j-1].dist; --j) std::swap(top_k[j], top_k[j-1]);
                    threshold = top_k[k-1].dist;
                }
            }
        }
#else
        size_t i = 0;
        for (; i < n; ++i) {
            int32_t distance = 0;
            for (int j = 0; j < 14; ++j) {
                int32_t d = (int32_t)q[j] - (int32_t)db_ptr[i].elements[j];
                distance += d * d;
            }
            if (distance < threshold) {
                if (count < k) {
                    top_k[count++] = {distance, (int)i};
                    if (count == k) {
                        std::sort(top_k, top_k + k, [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                        threshold = top_k[k-1].dist;
                    }
                } else {
                    top_k[k-1] = {distance, (int)i};
                    for (int j = k - 1; j > 0 && top_k[j].dist < top_k[j-1].dist; --j) std::swap(top_k[j], top_k[j-1]);
                    threshold = top_k[k-1].dist;
                }
            }
        }
#endif

#if defined(__x86_64__) || defined(_M_X64)
        for (; i < n; ++i) {
            __m256i v_ref = _mm256_load_si256((__m256i*)&db_ptr[i]);
            __m256i diff = _mm256_sub_epi16(v_query, v_ref);
            __m256i squared = _mm256_madd_epi16(diff, diff);
            __m128i low = _mm256_castsi256_si128(squared);
            __m128i high = _mm256_extracti128_si256(squared, 1);
            __m128i sum128 = _mm_add_epi32(low, high);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t distance = _mm_cvtsi128_si32(sum128);
            if (distance < threshold) {
                if (count < k) {
                    top_k[count++] = {distance, (int)i};
                    if (count == k) {
                        std::sort(top_k, top_k + k, [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                        threshold = top_k[k-1].dist;
                    }
                } else {
                    top_k[k-1] = {distance, (int)i};
                    for (int j = k - 1; j > 0 && top_k[j].dist < top_k[j-1].dist; --j) std::swap(top_k[j], top_k[j-1]);
                    threshold = top_k[k-1].dist;
                }
            }
        }
#endif

        std::vector<application::ports::out::SearchResult> results;
        results.reserve(count);
        for (int i = 0; i < count; ++i) {
            results.push_back({labels_[top_k[i].index] == 1, (float)top_k[i].dist});
        }
        return results;
    }

    size_t get_total_vectors() const { return database_.size(); }

    void save_binary(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        size_t n = database_.size();
        out.write((char*)&n, sizeof(n));
        out.write((char*)database_.data(), n * sizeof(PaddedVector));
        out.write((char*)labels_.data(), n * sizeof(int8_t));
    }

    bool load_binary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        size_t n;
        in.read((char*)&n, sizeof(n));
        database_.resize(n);
        labels_.resize(n);
        in.read((char*)database_.data(), n * sizeof(PaddedVector));
        in.read((char*)labels_.data(), n * sizeof(int8_t));
        return true;
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
