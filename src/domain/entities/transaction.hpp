#ifndef TRANSACTION_HPP
#define TRANSACTION_HPP

#include <string_view>
#include <vector>
#include <optional>
#include <array>

namespace domain {

// Optimization 3: Use std::string_view to avoid heap allocations
struct TransactionData {
    double amount;
    int installments;
    std::string_view requested_at;
};

struct Customer {
    double avg_amount;
    int tx_count_24h;
    std::vector<std::string_view> known_merchants;
};

struct Merchant {
    std::string_view id;
    std::string_view mcc;
    double avg_amount;
};

struct Terminal {
    bool is_online;
    bool card_present;
    double km_from_home;
};

struct LastTransaction {
    std::string_view timestamp;
    double km_from_current;
};

struct Transaction {
    std::string_view id;
    TransactionData transaction;
    Customer customer;
    Merchant merchant;
    Terminal terminal;
    std::optional<LastTransaction> last_transaction;
};

struct FraudResult {
    bool approved;
    double fraud_score;
};

using Vector14 = std::array<float, 14>;

} // namespace domain

#endif
