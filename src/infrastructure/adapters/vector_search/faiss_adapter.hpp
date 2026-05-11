#ifndef FAISS_ADAPTER_HPP
#define FAISS_ADAPTER_HPP

#include "../../../application/ports/out/vector_search_port.hpp"
#include "../../logging/logger.hpp"
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <memory>
#include <vector>
#include <fstream>
#include <omp.h>
#include <algorithm>

namespace infrastructure {
namespace adapters {
namespace vector_search {

class FaissAdapter : public application::ports::out::VectorSearchPort {
    std::unique_ptr<faiss::IndexIVFScalarQuantizer> index_;
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    std::vector<char> labels_;

public:
    FaissAdapter() {
        // Use all available cores for search
        omp_set_num_threads(std::max(1, (int)std::thread::hardware_concurrency()));
        
        quantizer_ = std::make_unique<faiss::IndexFlatL2>(14);
        
        // Balanced Performance: IVF4096 + SQ8
        // Fast enough for Rinha, accurate enough for fraud, memory efficient.
        index_ = std::make_unique<faiss::IndexIVFScalarQuantizer>(
            quantizer_.get(), 14, 4096, 
            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::METRIC_L2
        );
        
        index_->nprobe = 32; // Higher nprobe for better recall
    }

    void train(const std::vector<float>& vectors) {
        if (vectors.empty() || index_->is_trained) return;
        
        size_t n = vectors.size() / 14;
        logging::Logger::info("Training IVF index with " + std::to_string(n) + " vectors...");
        
        index_->train(n, vectors.data());
        
        if (!index_->is_trained) {
            logging::Logger::error("CRITICAL: IVF training failed!");
        } else {
            logging::Logger::info("IVF index training successful.");
        }
    }

    void add_batch(const std::vector<float>& vectors, const std::vector<char>& batch_labels) {
        if (vectors.empty() || !index_->is_trained) return;

        size_t n = vectors.size() / 14;
        index_->add(n, vectors.data());
        
        labels_.insert(labels_.end(), batch_labels.begin(), batch_labels.end());
    }

    size_t get_total_vectors() const {
        return index_ ? index_->ntotal : 0;
    }

    void save(const std::string& index_path, const std::string& labels_path) {
        if (index_) faiss::write_index(index_.get(), index_path.c_str());
        std::ofstream out(labels_path, std::ios::binary);
        if (out) out.write(labels_.data(), labels_.size());
    }

    bool load(const std::string& index_path, const std::string& labels_path) {
        try {
            faiss::Index* raw_index = faiss::read_index(index_path.c_str());
            if (!raw_index) return false;
            index_.reset(dynamic_cast<faiss::IndexIVFScalarQuantizer*>(raw_index));
            if (!index_) return false;
            index_->nprobe = 32;

            std::ifstream in(labels_path, std::ios::binary | std::ios::ate);
            if (in) {
                std::streamsize size = in.tellg();
                in.seekg(0, std::ios::beg);
                labels_.resize(size);
                in.read(labels_.data(), size);
            } else {
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    std::vector<application::ports::out::SearchResult> search(const domain::Vector14& query_vector, int k) override {
        if (!index_ || !index_->is_trained || index_->ntotal == 0) return {};

        std::vector<float> distances(k);
        std::vector<faiss::idx_t> indices(k);

        try {
            index_->search(1, query_vector.data(), k, distances.data(), indices.data());
        } catch (...) {
            return {};
        }

        std::vector<application::ports::out::SearchResult> results;
        for (int i = 0; i < k; ++i) {
            if (indices[i] >= 0 && indices[i] < static_cast<faiss::idx_t>(labels_.size())) {
                results.push_back({labels_[indices[i]] == 1, distances[i]});
            }
        }
        return results;
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
