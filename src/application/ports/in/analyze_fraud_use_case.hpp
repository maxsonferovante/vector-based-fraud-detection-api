#ifndef ANALYZE_FRAUD_USE_CASE_HPP
#define ANALYZE_FRAUD_USE_CASE_HPP

#include "../../../domain/entities/transaction.hpp"

namespace application {
namespace ports {
namespace in {

class AnalyzeFraudUseCase {
public:
    virtual ~AnalyzeFraudUseCase() = default;
    virtual domain::FraudResult execute(const domain::Transaction& tx, const domain::Customer& customer) = 0;
};

} // namespace in
} // namespace ports
} // namespace application

#endif
