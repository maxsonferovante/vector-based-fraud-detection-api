#include <boost/asio.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include "src/application/services/fraud_service.hpp"
#include "src/infrastructure/adapters/web/web_adapter.hpp"
#include "src/infrastructure/adapters/web/http_server.hpp"
#include "src/infrastructure/adapters/vector_search/simd_ivf_matcher.hpp"
#include "src/infrastructure/adapters/vector_search/data_loader.hpp"
#include "src/infrastructure/logging/logger.hpp"

using namespace infrastructure::adapters::vector_search;

int main(int argc, char* argv[]) {
    bool prepare_mode = (argc > 1 && std::string(argv[1]) == "--prepare");

    unsigned short port = 9999;
    auto const address = boost::asio::ip::make_address("0.0.0.0");

    const char* res_dir_env = std::getenv("RESOURCES_DIR");
    std::string res_dir = res_dir_env ? res_dir_env : "/app/resources";

    auto matcher = std::make_shared<SimdIvfMatcher>();

    domain::services::NormalizationConfig config;
    DataLoader::load_data(res_dir, matcher, config);
    matcher->apply_hugepages();
    matcher->log_memory_stats();

    if (prepare_mode) {
        DataLoader::save_binary_index(matcher, res_dir);
        infrastructure::logging::Logger::info("Preparation complete.");
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
