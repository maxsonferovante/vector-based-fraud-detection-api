#ifndef FRAUD_SERVICE_HPP
#define FRAUD_SERVICE_HPP

#include "../ports/in/analyze_fraud_use_case.hpp"
#include "../ports/out/vector_search_port.hpp"
#include "../../domain/services/normalizer.hpp"
#include "../../infrastructure/logging/logger.hpp"
#include <memory>
#include <sstream>
#include <iomanip>

namespace application {
namespace services {

class FraudService : public ports::in::AnalyzeFraudUseCase {
    std::shared_ptr<domain::services::Normalizer> normalizer_;
    std::shared_ptr<ports::out::VectorSearchPort> vector_search_;

public:
    FraudService(
        std::shared_ptr<domain::services::Normalizer> normalizer,
        std::shared_ptr<ports::out::VectorSearchPort> vector_search)
        : normalizer_(std::move(normalizer)), vector_search_(std::move(vector_search)) {}

    domain::FraudResult execute(const domain::Transaction& tx, const domain::Customer& customer) override {
        // Transform transaction into a feature vector
        auto vector = normalizer_->vectorize(tx);

        // Retrieve K nearest neighbors from the search engine
        auto neighbors = vector_search_->search(vector, 5);

        if (neighbors.count < 3) {
            return { true, 0.0 };
        }

        // Calculate score based on neighbor labels
        int fraud_count = 0;
        for (size_t i = 0; i < neighbors.count; ++i) {
            if (neighbors.items[i].is_fraud) {
                fraud_count++;
            }
        }

        // Com k=5, a transação é fraude se fraud_count >= 3.
        double fraud_score = static_cast<double>(fraud_count) / 5.0;
        bool approved = fraud_score < 0.6;

        return { approved, fraud_score };
    }
};

} // namespace services
} // namespace application

#endif
