#ifndef WEB_ADAPTER_HPP
#define WEB_ADAPTER_HPP

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <string_view>
#include <array>
#include <cmath>
#include <sstream>
#include "../../../application/ports/in/analyze_fraud_use_case.hpp"
#include "../../logging/logger.hpp"
#include "../../parser/json_parser.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
using tcp = boost::asio::ip::tcp;
using FastParser = infrastructure::parser::FastJsonScanner;

namespace infrastructure {
namespace adapters {
namespace web {

class WebAdapter {
    std::shared_ptr<application::ports::in::AnalyzeFraudUseCase> use_case_;

    static constexpr std::string_view RESP_READY = "{\"service\":\"fraud-detection-api\",\"status\":\"ready\"}";
    static constexpr std::string_view RESP_NOT_FOUND = "{\"error\":\"Not Found\"}";
    static constexpr std::string_view RESP_BAD_REQUEST = "{\"error\":\"Invalid Request\"}";
    
    static constexpr std::array<std::string_view, 6> RESP_FRAUD = {
        "{\"approved\":true,\"fraud_score\":0.0}",
        "{\"approved\":true,\"fraud_score\":0.2}",
        "{\"approved\":true,\"fraud_score\":0.4}",
        "{\"approved\":false,\"fraud_score\":0.6}",
        "{\"approved\":false,\"fraud_score\":0.8}",
        "{\"approved\":false,\"fraud_score\":1.0}"
    };

public:
    explicit WebAdapter(std::shared_ptr<application::ports::in::AnalyzeFraudUseCase> use_case)
        : use_case_(std::move(use_case)) {}

    http::response<http::string_body> handle_request(tcp::endpoint remote, http::request<http::string_body> const& req) {
        if (req.method() == http::verb::get && req.target() == "/ready") {
            return make_static_response(http::status::ok, req.version(), req.keep_alive(), RESP_READY);
        } 
        
        if (req.method() == http::verb::post && req.target() == "/fraud-score") {
            return process_fraud_score(req);
        }

        return make_static_response(http::status::not_found, req.version(), req.keep_alive(), RESP_NOT_FOUND);
    }

private:
    http::response<http::string_body> process_fraud_score(http::request<http::string_body> const& req) {
        try {
            std::string_view body = req.body();
            
            domain::Transaction tx;
            tx.id = FastParser::find_string(body, FastParser::K_ID);
            
            std::string_view j_tx = FastParser::find_object(body, FastParser::K_TX);
            tx.transaction = {
                FastParser::find_double(j_tx, FastParser::K_AMOUNT), 
                FastParser::find_int(j_tx, FastParser::K_INST), 
                FastParser::find_string(j_tx, FastParser::K_REQ)
            };
            
            std::string_view j_cust = FastParser::find_object(body, FastParser::K_CUST);
            tx.customer.avg_amount = FastParser::find_double(j_cust, FastParser::K_AVG);
            tx.customer.tx_count_24h = FastParser::find_int(j_cust, FastParser::K_COUNT);
            
            //conhecidos merchants (extração básica simplificada)
            size_t m_start = j_cust.find("[");
            size_t m_end = j_cust.find("]");
            if (m_start != std::string_view::npos && m_end != std::string_view::npos && m_end > m_start) {
                std::string_view list = j_cust.substr(m_start + 1, m_end - m_start - 1);
                size_t current = 0;
                while (current < list.size()) {
                    size_t s = list.find("\"", current);
                    if (s == std::string_view::npos) break;
                    size_t e = list.find("\"", s + 1);
                    if (e == std::string_view::npos) break;
                    tx.customer.known_merchants.push_back(list.substr(s + 1, e - s - 1));
                    current = e + 1;
                }
            }
            
            std::string_view j_merch = FastParser::find_object(body, FastParser::K_MERCH);
            tx.merchant = {
                FastParser::find_string(j_merch, FastParser::K_ID), 
                FastParser::find_string(j_merch, FastParser::K_MCC), 
                FastParser::find_double(j_merch, FastParser::K_AVG)
            };
            
            std::string_view j_term = FastParser::find_object(body, FastParser::K_TERM);
            tx.terminal = {
                FastParser::find_bool(j_term, FastParser::K_ONLINE), 
                FastParser::find_bool(j_term, FastParser::K_CARD), 
                FastParser::find_double(j_term, FastParser::K_HOME)
            };
            
            std::string_view j_last = FastParser::find_object(body, FastParser::K_LAST);
            if (!j_last.empty()) {
                tx.last_transaction = domain::LastTransaction{
                    FastParser::find_string(j_last, FastParser::K_TS), 
                    FastParser::find_double(j_last, FastParser::K_KM)
                };
            }

            auto result = use_case_->execute(tx, tx.customer);

            int idx = static_cast<int>(std::round(result.fraud_score * 5.0));
            idx = std::clamp(idx, 0, 5);

            return make_static_response(http::status::ok, req.version(), req.keep_alive(), RESP_FRAUD[idx]);

        } catch (...) {
            return make_static_response(http::status::bad_request, req.version(), req.keep_alive(), RESP_BAD_REQUEST);
        }
    }

    http::response<http::string_body> make_static_response(http::status status, unsigned version, bool keep_alive, std::string_view body) {
        http::response<http::string_body> res{status, version};
        res.set(http::field::server, "Rinha API");
        res.set(http::field::content_type, "application/json");
        res.keep_alive(keep_alive);
        res.body() = std::string(body);
        res.prepare_payload();
        return res;
    }
};

} // namespace web
} // namespace adapters
} // namespace infrastructure

#endif
