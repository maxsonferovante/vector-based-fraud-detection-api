#include <memory>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include "src/application/services/fraud_service.hpp"
#include "src/infrastructure/adapters/web/web_adapter.hpp"
#include "src/infrastructure/adapters/web/http_server.hpp"
#include "src/infrastructure/adapters/vector_search/simd_ivf_matcher.hpp"
#include "src/infrastructure/adapters/vector_search/data_loader.hpp"
#include "src/infrastructure/logging/logger.hpp"

using namespace infrastructure::adapters::vector_search;

int main(int argc, char* argv[]) {
    try {
        bool prepare_mode = (argc > 1 && std::string(argv[1]) == "--prepare");

        unsigned short port = 9999;
        auto const address = boost::asio::ip::make_address("0.0.0.0");

        const char* res_dir_env = std::getenv("RESOURCES_DIR");
        std::string res_dir = res_dir_env ? res_dir_env : "/app/resources";

        // Initialize vector search engine
        auto matcher = std::make_shared<SimdIvfMatcher>();

        // Load configuration and reference data
        domain::services::NormalizationConfig config;
        DataLoader::load_data(res_dir, matcher, config);

        if (prepare_mode) {
            DataLoader::save_binary_index(matcher, res_dir);
            infrastructure::logging::Logger::info("Preparation complete.");
            return 0;
        }

        if (matcher->get_total_vectors() == 0) {
            infrastructure::logging::Logger::error("CRITICAL: No data loaded. Aborting.");
            return 1;
        }

        // Initialize domain services
        auto normalizer = std::make_shared<domain::services::Normalizer>(config);
        auto fraud_service = std::make_shared<application::services::FraudService>(normalizer, matcher);

        // Initialize web infrastructure
        auto web_adapter = std::make_shared<infrastructure::adapters::web::WebAdapter>(fraud_service);

        // Start HTTP server with a single processing thread
        int threads = 1;
        boost::asio::io_context ioc{threads};
        auto listener = std::make_shared<infrastructure::adapters::web::Listener>(
            ioc, 
            tcp::endpoint{address, port}, 
            web_adapter
        );

        infrastructure::logging::Logger::info("API service started on port " + std::to_string(port));
        
        listener->run();
        ioc.run();
    } catch (const std::exception& e) {
        infrastructure::logging::Logger::error(std::string("Critical error: ") + e.what());
        return 1;
    }

    return 0;
}
