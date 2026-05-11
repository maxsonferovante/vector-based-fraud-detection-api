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

        // 1. Inicializar Motor de Busca (SIMD IVF)
        auto matcher = std::make_shared<SimdIvfMatcher>();

        // 2. Carregar dados e configurações
        domain::services::NormalizationConfig config;
        DataLoader::load_data(res_dir, matcher, config);

        if (prepare_mode) {
            DataLoader::save_binary_index(matcher, res_dir);
            infrastructure::logging::Logger::info("Preparation complete.");
            return 0;
        }

        // FAIL-FAST
        if (matcher->get_total_vectors() == 0) {
            infrastructure::logging::Logger::error("CRITICAL: No data loaded. Aborting.");
            return 1;
        }

        // 3. Inicializar Normalizador
        auto normalizer = std::make_shared<domain::services::Normalizer>(config);

        // 4. Inicializar Serviço de Fraude
        auto fraud_service = std::make_shared<application::services::FraudService>(normalizer, matcher);

        // 5. Inicializar Adaptador Web
        auto web_adapter = std::make_shared<infrastructure::adapters::web::WebAdapter>(fraud_service);

        // 6. Iniciar Servidor HTTP
        // Com SIMD IVF, 1 thread é o ideal para 0.45 CPU
        int threads = 1;
        boost::asio::io_context ioc{threads};
        auto listener = std::make_shared<infrastructure::adapters::web::Listener>(
            ioc, 
            tcp::endpoint{address, port}, 
            web_adapter
        );

        infrastructure::logging::Logger::info("Rinha API started on port " + std::to_string(port) + " with 1 thread");
        
        listener->run();

        ioc.run();
    } catch (const std::exception& e) {
        infrastructure::logging::Logger::error(std::string("Critical error: ") + e.what());
        return 1;
    }

    return 0;
}
