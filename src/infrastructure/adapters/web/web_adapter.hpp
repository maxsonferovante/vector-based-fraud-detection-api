#ifndef WEB_ADAPTER_HPP
#define WEB_ADAPTER_HPP

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <string_view>
#include <array>
#include <cmath>
#include "../../../application/ports/in/analyze_fraud_use_case.hpp"
#include "../../logging/logger.hpp"
#include "../../parser/simdjson_request_parser.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
using tcp = boost::asio::ip::tcp;
using SimdjsonParser = infrastructure::parser::SimdjsonRequestParser;

namespace infrastructure {
namespace adapters {
namespace web {

class WebAdapter {
    std::shared_ptr<application::ports::in::AnalyzeFraudUseCase> use_case_;

    // Respostas pré-computadas como strings estáticas — zero alocação no hot path.
    // fraud_score só pode ser 0/5, 1/5, 2/5, 3/5, 4/5, 5/5 → 6 respostas fixas.
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

    // Headers constantes pré-setados; apenas body e status mudam por request.
    static constexpr std::string_view CT_JSON = "application/json";
    static constexpr std::string_view SERVER_NAME = "Rinha API";

public:
    explicit WebAdapter(std::shared_ptr<application::ports::in::AnalyzeFraudUseCase> use_case)
        : use_case_(std::move(use_case)) {}

    http::response<http::string_body> handle_request(tcp::endpoint remote, http::request<http::string_body> const& req) {
        // Fast path: check target length first to avoid string comparison
        const auto target = req.target();
        
        if (req.method() == http::verb::post && target.size() == 12 && target == "/fraud-score") {
            return process_fraud_score(req);
        }

        if (req.method() == http::verb::get && target.size() == 6 && target == "/ready") {
            return make_response(http::status::ok, req.version(), req.keep_alive(), RESP_READY);
        }

        return make_response(http::status::not_found, req.version(), req.keep_alive(), RESP_NOT_FOUND);
    }

private:
    http::response<http::string_body> process_fraud_score(http::request<http::string_body> const& req) {
        domain::Transaction tx{};
        
        // simdjson ondemand parser — SIMD-acelerado, thread-local
        if (!SimdjsonParser::parse(req.body(), tx)) {
            return make_response(http::status::bad_request, req.version(), req.keep_alive(), RESP_BAD_REQUEST);
        }

        auto result = use_case_->execute(tx, tx.customer);

        // Mapeia fraud_score para índice [0,5]: fraud_count/5 * 5 = fraud_count
        int idx = static_cast<int>(std::round(result.fraud_score * 5.0));
        idx = std::clamp(idx, 0, 5);

        return make_response(http::status::ok, req.version(), req.keep_alive(), RESP_FRAUD[idx]);
    }

    http::response<http::string_body> make_response(http::status status, unsigned version, 
                                                     bool keep_alive, std::string_view body) {
        http::response<http::string_body> res{status, version};
        res.set(http::field::server, SERVER_NAME);
        res.set(http::field::content_type, CT_JSON);
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
