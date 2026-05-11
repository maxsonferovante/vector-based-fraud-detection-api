#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <zlib.h>
#include <filesystem>
#include <memory>
#include "../../logging/logger.hpp"
#include "../../../domain/services/normalizer.hpp"
#include "../../parser/json_parser.hpp"
#include "faiss_adapter.hpp"

namespace infrastructure {
namespace adapters {
namespace vector_search {

namespace fs = std::filesystem;
using FastParser = infrastructure::parser::FastJsonScanner;

class DataLoader {
public:
    static void save_binary_index(std::shared_ptr<FaissAdapter> faiss_adapter, const std::string& base_dir) {
        std::string index_path = base_dir + "/faiss_index.bin";
        std::string labels_path = base_dir + "/labels.bin";
        faiss_adapter->save(index_path, labels_path);
        logging::Logger::info("Saved binary FAISS index to " + base_dir);
    }

    // Optimization 1: STREAMING and BATCHING to save RAM
    static void load_data(
        const std::string& base_dir,
        std::shared_ptr<FaissAdapter> faiss_adapter,
        domain::services::NormalizationConfig& out_config) 
    {
        std::string norm_path = base_dir + "/normalization.json";
        std::string mcc_path = base_dir + "/mcc_risk.json";
        std::string ref_path = base_dir + "/references.json.gz";

        logging::Logger::info("Loading resources in STREAMING mode from " + base_dir);

        // 1. Normalização (Arquivo pequeno, leitura direta)
        std::ifstream norm_file(norm_path);
        if (norm_file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(norm_file)), std::istreambuf_iterator<char>());
            out_config.max_amount = FastParser::find_double(content, "max_amount");
            out_config.max_installments = FastParser::find_double(content, "max_installments");
            out_config.amount_vs_avg_ratio = FastParser::find_double(content, "amount_vs_avg_ratio");
            out_config.max_minutes = FastParser::find_double(content, "max_minutes");
            out_config.max_km = FastParser::find_double(content, "max_km");
            out_config.max_tx_count_24h = FastParser::find_double(content, "max_tx_count_24h");
            out_config.max_merchant_avg_amount = FastParser::find_double(content, "max_merchant_avg_amount");
        }

        // 2. MCC (Arquivo pequeno)
        std::ifstream mcc_file(mcc_path);
        if (mcc_file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(mcc_file)), std::istreambuf_iterator<char>());
            size_t current = 0;
            while (true) {
                size_t s = content.find("\"", current);
                if (s == std::string::npos) break;
                size_t e = content.find("\"", s + 1);
                std::string key(content.substr(s + 1, e - s - 1));
                size_t val_start = content.find_first_of("0123456789.", content.find(":", e));
                size_t val_end = content.find_first_not_of("0123456789.", val_start);
                try {
                    out_config.mcc_risk[key] = std::stod(std::string(content.substr(val_start, val_end - val_start)));
                } catch(...) {}
                current = (val_end == std::string::npos) ? content.size() : val_end;
            }
        }

        // 3. Referências (O GRANDE DESAFIO: 3M registros / ~300MB)
        std::string index_path = base_dir + "/faiss_index.bin";
        std::string labels_path = base_dir + "/labels.bin";

        if (fs::exists(index_path) && fs::exists(labels_path)) {
            logging::Logger::info("Loading pre-computed FAISS index from " + base_dir);
            if (faiss_adapter->load(index_path, labels_path)) {
                logging::Logger::info("Successfully loaded " + std::to_string(faiss_adapter->get_total_vectors()) + " reference vectors from binary.");
                return;
            }
            logging::Logger::error("Failed to load binary index, falling back to JSON stream.");
        }

        gzFile file = gzopen(ref_path.c_str(), "rb");
        if (!file) {
            logging::Logger::error("CRITICAL: Could not open " + ref_path);
            return;
        }

        // Buffer de leitura para o stream descompactado
        std::string window; 
        window.reserve(128 * 1024); // 128KB de janela de processamento
        
        char buffer[32768]; // Buffer de descompactação
        int bytes_read;
        
        logging::Logger::info("Parsing 3M vectors via stream in batches...");
        
        std::vector<float> batch_vectors;
        std::vector<char> batch_labels;
        const size_t BATCH_SIZE = 100000;
        batch_vectors.reserve(BATCH_SIZE * 14);
        batch_labels.reserve(BATCH_SIZE);
        
        bool is_trained = false;

        while ((bytes_read = gzread(file, buffer, sizeof(buffer))) > 0) {
            window.append(buffer, bytes_read);
            
            size_t start = 0;
            while (true) {
                size_t obj_start = window.find("{", start);
                if (obj_start == std::string_view::npos) break;
                
                size_t obj_end = window.find("}", obj_start);
                if (obj_end == std::string_view::npos) {
                    // Objeto incompleto na janela, espera o próximo gzread
                    break; 
                }
                
                std::string_view obj = std::string_view(window).substr(obj_start, obj_end - obj_start + 1);
                std::vector<float> v = FastParser::extract_vector(obj);
                
                if (v.size() == 14) {
                    batch_vectors.insert(batch_vectors.end(), v.begin(), v.end());
                    std::string_view label = FastParser::find_string(obj, "label");
                    batch_labels.push_back(label == "fraud" ? 1 : 0);
                    
                    if (batch_labels.size() >= BATCH_SIZE) {
                        if (!is_trained) {
                            faiss_adapter->train(batch_vectors);
                            is_trained = true;
                        }
                        faiss_adapter->add_batch(batch_vectors, batch_labels);
                        batch_vectors.clear();
                        batch_labels.clear();
                    }
                }
                
                start = obj_end + 1;
            }
            
            // Remove o que já foi processado da janela para economizar RAM
            if (start > 0) {
                window.erase(0, start);
            }
        }
        
        // Adicionar batch final restante
        if (!batch_labels.empty()) {
            if (!is_trained) {
                faiss_adapter->train(batch_vectors);
            }
            faiss_adapter->add_batch(batch_vectors, batch_labels);
        }
        
        batch_vectors.clear();
        batch_vectors.shrink_to_fit();
        batch_labels.clear();
        batch_labels.shrink_to_fit();
        
        gzclose(file);
        logging::Logger::info("Successfully loaded " + std::to_string(faiss_adapter->get_total_vectors()) + " reference vectors.");
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
