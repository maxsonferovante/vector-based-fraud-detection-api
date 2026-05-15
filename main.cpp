#include <boost/asio.hpp>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <random>
#include "src/application/services/fraud_service.hpp"
#include "src/infrastructure/adapters/web/web_adapter.hpp"
#include "src/infrastructure/adapters/web/http_server.hpp"
#include "src/infrastructure/adapters/vector_search/simd_ivf_matcher.hpp"
#include "src/infrastructure/adapters/vector_search/data_loader.hpp"
#include "src/infrastructure/logging/logger.hpp"

using namespace infrastructure::adapters::vector_search;
using tcp = boost::asio::ip::tcp;

static void run_pgo_warmup(
    std::shared_ptr<SimdIvfMatcher> matcher,
    std::shared_ptr<domain::services::Normalizer> normalizer)
{
    infrastructure::logging::Logger::info("PGO warm-up: running synthetic queries...");

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);
    std::uniform_int_distribution<int>    dist_bool(0, 1);

    // dead code durante a instrumentação PGO.
    volatile int fraud_count = 0;
    volatile int legit_count = 0;

    constexpr int WARMUP_QUERIES = 10000;
    infrastructure::logging::Logger::info("Running " + std::to_string(WARMUP_QUERIES) + " warm-up queries...");
    for (int i = 0; i < WARMUP_QUERIES; ++i) {
        domain::Vector14 q;
        bool use_sentinel = (i % 2 == 0);
        for (int j = 0; j < 14; ++j) {
            if (use_sentinel && (j == 5 || j == 6)) {
                q[j] = -1.0f;
            } else {
                q[j] = dist_01(rng);
            }
        }
        auto result = matcher->search(q, 5);

        // Conta resultados para impedir que o compilador elimine como dead code.
        for (size_t r = 0; r < result.count; ++r) {
            if (result.items[r].is_fraud) ++fraud_count;
            else ++legit_count;
        }
    }

    infrastructure::logging::Logger::info(
        "PGO warm-up done: " + std::to_string(WARMUP_QUERIES) + " queries, " +
        "fraud_hits=" + std::to_string(fraud_count) +
        " legit_hits=" + std::to_string(legit_count));
}

int main(int argc, char* argv[]) {
    bool prepare_mode = (argc > 1 && std::string(argv[1]) == "--prepare");

    unsigned short port = 9999;
    auto const address = boost::asio::ip::make_address("0.0.0.0");

    const char* res_dir_env = std::getenv("RESOURCES_DIR");
    std::string res_dir = res_dir_env ? res_dir_env : "/app/resources";

    auto matcher = std::make_shared<SimdIvfMatcher>();

    domain::services::NormalizationConfig config;
    DataLoader::load_data(res_dir, matcher, config);

    if (prepare_mode) {
        // Salva o índice binário primeiro.
        DataLoader::save_binary_index(matcher, res_dir);

        // Roda o warm-up PGO para coletar perfil do hot path real.
        // O normalizer é instanciado aqui apenas para o warm-up.
        auto normalizer = std::make_shared<domain::services::Normalizer>(config);
        run_pgo_warmup(matcher, normalizer);

        infrastructure::logging::Logger::info("Preparation complete (index + PGO profile).");
        return 0;
    }

    if (matcher->get_total_vectors() == 0) {
        infrastructure::logging::Logger::error("CRITICAL: No data loaded. Aborting.");
        return 1;
    }

    auto normalizer    = std::make_shared<domain::services::Normalizer>(config);
    auto fraud_service = std::make_shared<application::services::FraudService>(normalizer, matcher);
    auto web_adapter   = std::make_shared<infrastructure::adapters::web::WebAdapter>(fraud_service);

    const int num_threads = std::clamp(
        static_cast<int>(std::thread::hardware_concurrency()), 1, 2);

    infrastructure::logging::Logger::info(
        "Starting on port " + std::to_string(port) +
        " threads=" + std::to_string(num_threads));

    boost::asio::io_context ioc{num_threads};
    auto listener = std::make_shared<infrastructure::adapters::web::Listener>(
        ioc, tcp::endpoint{address, port}, web_adapter);
    listener->run();

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(num_threads - 1));
    for (int i = 0; i < num_threads - 1; ++i)
        pool.emplace_back([&ioc]{ ioc.run(); });
    ioc.run();

    for (auto& t : pool) t.join();

    return 0;
}