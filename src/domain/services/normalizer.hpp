#ifndef NORMALIZER_HPP
#define NORMALIZER_HPP

#include "../entities/transaction.hpp"
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <array>
#include <string_view>
#include <string>
#include <ctime>

namespace domain {
namespace services {

// Heterogeneous hash for unordered_map lookup with string_view
struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view sv) const {
        return std::hash<std::string_view>{}(sv);
    }
};

struct NormalizationConfig {
    double max_amount = 10000.0;
    double max_installments = 12.0;
    double amount_vs_avg_ratio = 10.0;
    double max_minutes = 1440.0;
    double max_km = 1000.0;
    double max_tx_count_24h = 20.0;
    double max_merchant_avg_amount = 10000.0;
    // Optimization 5: O(1) heterogeneous lookup without allocations
    std::unordered_map<std::string, double, StringHash, std::equal_to<>> mcc_risk;
};

struct LUT {
    static constexpr std::array<float, 24> hours = [] {
        std::array<float, 24> a{};
        for (int i = 0; i < 24; ++i) a[i] = static_cast<float>(i) / 23.0f;
        return a;
    }();

    static constexpr std::array<float, 7> days = [] {
        std::array<float, 7> a{};
        for (int i = 0; i < 7; ++i) a[i] = static_cast<float>(i) / 6.0f;
        return a;
    }();
};

class Normalizer {
    NormalizationConfig config_;
    double inv_max_amount;
    double inv_max_installments;
    double inv_amount_vs_avg_ratio;
    double inv_max_minutes;
    double inv_max_km;
    double inv_max_tx_count_24h;
    double inv_max_merchant_avg_amount;

public:
    explicit Normalizer(NormalizationConfig config) : config_(std::move(config)) {
        inv_max_amount = 1.0 / config_.max_amount;
        inv_max_installments = 1.0 / config_.max_installments;
        inv_amount_vs_avg_ratio = 1.0 / config_.amount_vs_avg_ratio;
        inv_max_minutes = 1.0 / config_.max_minutes;
        inv_max_km = 1.0 / config_.max_km;
        inv_max_tx_count_24h = 1.0 / config_.max_tx_count_24h;
        inv_max_merchant_avg_amount = 1.0 / config_.max_merchant_avg_amount;
    }

    std::array<float, 14> vectorize(const Transaction& tx) const {
        std::array<float, 14> v;

        v[0] = clamp(tx.transaction.amount * inv_max_amount);
        v[1] = clamp(static_cast<double>(tx.transaction.installments) * inv_max_installments);
        
        // Proteção contra divisão por zero
        double avg = (tx.customer.avg_amount > 0) ? tx.customer.avg_amount : 1.0;
        v[2] = clamp((tx.transaction.amount / avg) * inv_amount_vs_avg_ratio);

        auto [hour, day_of_week, ts_unix] = parse_date_fast(tx.transaction.requested_at);
        
        v[3] = LUT::hours[hour % 24];
        v[4] = LUT::days[day_of_week % 7];

        if (tx.last_transaction.has_value()) {
            auto [_, __, last_ts_unix] = parse_date_fast(tx.last_transaction->timestamp);
            double diff_minutes = std::abs(static_cast<double>(ts_unix - last_ts_unix)) / 60.0;

            v[5] = clamp(diff_minutes * inv_max_minutes);
            v[6] = clamp(tx.last_transaction->km_from_current * inv_max_km);
        } else {
            v[5] = -1.0f;
            v[6] = -1.0f;
        }

        v[7] = clamp(tx.terminal.km_from_home * inv_max_km);
        v[8] = clamp(static_cast<double>(tx.customer.tx_count_24h) * inv_max_tx_count_24h);
        v[9] = tx.terminal.is_online ? 1.0f : 0.0f;
        v[10] = tx.terminal.card_present ? 1.0f : 0.0f;

        bool known = false;
        for (const auto& m_id : tx.customer.known_merchants) {
            if (m_id == tx.merchant.id) {
                known = true;
                break;
            }
        }
        v[11] = known ? 0.0f : 1.0f;

        auto it = config_.mcc_risk.find(tx.merchant.mcc);
        v[12] = (it != config_.mcc_risk.end()) ? static_cast<float>(it->second) : 0.5f;
        v[13] = clamp(tx.merchant.avg_amount * inv_max_merchant_avg_amount);

        return v;
    }

private:
    static float clamp(double val) {
        if (val < 0.0) return 0.0f;
        if (val > 1.0) return 1.0f;
        return static_cast<float>(val);
    }

    struct FastDateResult {
        int hour;
        int day_of_week;
        int64_t unix_timestamp;
    };

    // Optimization 6: Pure mathematical date parsing bypassing libc
    static FastDateResult parse_date_fast(std::string_view s) {
        if (s.size() < 19) return {0, 0, 0};

        int y = (s[0]-'0')*1000 + (s[1]-'0')*100 + (s[2]-'0')*10 + (s[3]-'0');
        int m = (s[5]-'0')*10 + (s[6]-'0');
        int d = (s[8]-'0')*10 + (s[9]-'0');
        int h = (s[11]-'0')*10 + (s[12]-'0');
        int min = (s[14]-'0')*10 + (s[15]-'0');
        int sec = (s[17]-'0')*10 + (s[18]-'0');

        // Unix timestamp calculation
        int yy = y - (m <= 2 ? 1 : 0);
        int mm = (m <= 2 ? m + 12 : m);
        int days = 365 * yy + yy / 4 - yy / 100 + yy / 400 + 306 * (mm + 1) / 10 + d - 719528;
        int64_t ts = static_cast<int64_t>(days) * 86400 + h * 3600 + min * 60 + sec;

        // Day of week calculation (0=Monday, 6=Sunday for Rinha specs)
        // Jan 1, 1970 was Thursday.
        int wday = (days + 3) % 7;
        if (wday < 0) wday += 7; // Ensures positive modulo
        // Mapping: 0=Mon..6=Sun
        // Standard wday gives 0=Mon, 1=Tue...6=Sun based on days offset (epoch offset handles it)

        return {h, wday, ts};
    }
};

} // namespace services
} // namespace domain

#endif
