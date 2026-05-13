#ifndef SIMDJSON_REQUEST_PARSER_HPP
#define SIMDJSON_REQUEST_PARSER_HPP

#include <simdjson.h>
#include <string_view>
#include <array>
#include <cstring>
#include "../../domain/entities/transaction.hpp"

namespace infrastructure {
namespace parser {

/**
 * Parser de requests HTTP usando simdjson ondemand.
 * Thread-local para evitar contention; scratch buffer pré-alocado.
 * simdjson requer SIMDJSON_PADDING bytes extras após o input.
 */
class SimdjsonRequestParser {
    static constexpr size_t SCRATCH_SIZE = 8 * 1024;
    static constexpr size_t PADDING = simdjson::SIMDJSON_PADDING;

    struct ThreadCtx {
        alignas(64) std::array<char, SCRATCH_SIZE + PADDING> scratch{};
        simdjson::ondemand::parser parser{SCRATCH_SIZE};
    };

    static ThreadCtx& ctx() noexcept {
        thread_local ThreadCtx c;
        return c;
    }

    // ISO 8601 date parsing — idêntico ao Normalizer::parse_date_fast
    struct DateInfo {
        int hour;
        int day_of_week;
        int64_t unix_timestamp;
    };

    static DateInfo parse_iso(std::string_view s) noexcept {
        if (s.size() < 19) return {0, 0, 0};

        int y = (s[0]-'0')*1000 + (s[1]-'0')*100 + (s[2]-'0')*10 + (s[3]-'0');
        int m = (s[5]-'0')*10 + (s[6]-'0');
        int d = (s[8]-'0')*10 + (s[9]-'0');
        int h = (s[11]-'0')*10 + (s[12]-'0');
        int min_val = (s[14]-'0')*10 + (s[15]-'0');
        int sec = (s[17]-'0')*10 + (s[18]-'0');

        int yy = y - (m <= 2 ? 1 : 0);
        int mm = (m <= 2 ? m + 12 : m);
        int days = 365 * yy + yy / 4 - yy / 100 + yy / 400 + 306 * (mm + 1) / 10 + d - 719528;
        int64_t ts = static_cast<int64_t>(days) * 86400 + h * 3600 + min_val * 60 + sec;

        int wday = (days + 3) % 7;
        if (wday < 0) wday += 7;

        return {h, wday, ts};
    }

public:
    /**
     * Parseia o body do request diretamente em um Transaction.
     * Retorna true em caso de sucesso.
     * O Transaction retornado mantém string_views no scratch buffer —
     * válido apenas até a próxima chamada de parse().
     */
    static bool parse(std::string_view body, domain::Transaction& tx) noexcept {
        if (body.empty() || body.size() >= SCRATCH_SIZE) return false;

        auto& c = ctx();
        std::memcpy(c.scratch.data(), body.data(), body.size());
        std::memset(c.scratch.data() + body.size(), 0, PADDING);

        simdjson::ondemand::document doc;
        if (c.parser.iterate(c.scratch.data(), body.size(), c.scratch.size())
                .get(doc) != simdjson::SUCCESS) {
            return false;
        }

        // -- transaction --
        {
            auto tx_res = doc.find_field_unordered("transaction");
            if (tx_res.error() != simdjson::SUCCESS) return false;
            simdjson::ondemand::object tx_obj = tx_res.value_unsafe();

            double amount = 0.0;
            if (tx_obj["amount"].get(amount) != simdjson::SUCCESS) return false;
            tx.transaction.amount = amount;

            int64_t inst = 0;
            if (tx_obj["installments"].get(inst) != simdjson::SUCCESS) return false;
            tx.transaction.installments = static_cast<int>(inst);

            std::string_view req_sv;
            if (tx_obj["requested_at"].get_string().get(req_sv) != simdjson::SUCCESS) return false;
            tx.transaction.requested_at = req_sv;
        }

        // -- customer --
        {
            auto cust_res = doc.find_field_unordered("customer");
            if (cust_res.error() != simdjson::SUCCESS) return false;
            simdjson::ondemand::object cust = cust_res.value_unsafe();

            double avg = 0.0;
            if (cust["avg_amount"].get(avg) != simdjson::SUCCESS) return false;
            tx.customer.avg_amount = avg;

            int64_t cnt = 0;
            if (cust["tx_count_24h"].get(cnt) != simdjson::SUCCESS) return false;
            tx.customer.tx_count_24h = static_cast<int>(cnt);

            tx.customer.known_merchants_count = 0;
            auto km_field = cust["known_merchants"];
            if (km_field.error() == simdjson::SUCCESS) {
                simdjson::ondemand::array arr = km_field.value_unsafe();
                for (auto elem : arr) {
                    std::string_view sv;
                    if (elem.get_string().get(sv) != simdjson::SUCCESS) continue;
                    if (tx.customer.known_merchants_count < static_cast<int>(tx.customer.known_merchants.size())) {
                        tx.customer.known_merchants[tx.customer.known_merchants_count++] = sv;
                    }
                }
            }
        }

        // -- merchant --
        {
            auto m_res = doc.find_field_unordered("merchant");
            if (m_res.error() != simdjson::SUCCESS) return false;
            simdjson::ondemand::object m = m_res.value_unsafe();

            std::string_view mid;
            if (m["id"].get_string().get(mid) == simdjson::SUCCESS) {
                tx.merchant.id = mid;
            }

            std::string_view mcc_sv;
            if (m["mcc"].get_string().get(mcc_sv) == simdjson::SUCCESS) {
                tx.merchant.mcc = mcc_sv;
            }

            double mavg = 0.0;
            if (m["avg_amount"].get(mavg) != simdjson::SUCCESS) return false;
            tx.merchant.avg_amount = mavg;
        }

        // -- terminal --
        {
            auto t_res = doc.find_field_unordered("terminal");
            if (t_res.error() != simdjson::SUCCESS) return false;
            simdjson::ondemand::object t = t_res.value_unsafe();

            bool b = false;
            if (t["is_online"].get(b) != simdjson::SUCCESS) return false;
            tx.terminal.is_online = b;

            if (t["card_present"].get(b) != simdjson::SUCCESS) return false;
            tx.terminal.card_present = b;

            double km = 0.0;
            if (t["km_from_home"].get(km) != simdjson::SUCCESS) return false;
            tx.terminal.km_from_home = km;
        }

        // -- last_transaction (nullable) --
        {
            auto lt_field = doc.find_field_unordered("last_transaction");
            if (lt_field.error() == simdjson::SUCCESS) {
                simdjson::ondemand::value lt = lt_field.value_unsafe();
                if (lt.is_null().value_unsafe()) {
                    tx.last_transaction = std::nullopt;
                } else {
                    std::string_view ts_sv;
                    if (lt["timestamp"].get_string().get(ts_sv) != simdjson::SUCCESS) return false;

                    double km = 0.0;
                    if (lt["km_from_current"].get(km) != simdjson::SUCCESS) return false;

                    tx.last_transaction = domain::LastTransaction{ts_sv, km};
                }
            } else {
                tx.last_transaction = std::nullopt;
            }
        }

        return true;
    }
};

} // namespace parser
} // namespace infrastructure

#endif
