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
#include "simd_ivf_matcher.hpp"

namespace infrastructure {
namespace adapters {
namespace vector_search {

namespace fs = std::filesystem;
using FastParser = infrastructure::parser::FastJsonScanner;

class DataLoader {
public:
    static void save_binary_index(std::shared_ptr<SimdIvfMatcher> matcher, const std::string& base_dir) {
        std::string path = base_dir + "/matcher.bin";
        matcher->save_binary(path);
        logging::Logger::info("Index saved to " + base_dir);
    }

    static void load_data(
        const std::string& base_dir,
        std::shared_ptr<SimdIvfMatcher> matcher,
        domain::services::NormalizationConfig& out_config) 
    {
        std::string norm_path = base_dir + "/normalization.json";
        std::string mcc_path = base_dir + "/mcc_risk.json";
        std::string ref_path = base_dir + "/references.json.gz";

        logging::Logger::info("Loading configuration resources...");

        // 1. Normalization Config
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

        // 2. MCC Risk Mapping
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

        // 3. Vector Database
        std::string bin_path = base_dir + "/matcher.bin";
        if (fs::exists(bin_path)) {
            logging::Logger::info("Loading binary index from " + bin_path);
            if (matcher->load_binary(bin_path)) {
                return;
            }
        }

        gzFile file = gzopen(ref_path.c_str(), "rb");
        if (!file) {
            logging::Logger::error("CRITICAL: Could not open dataset " + ref_path);
            return;
        }

        std::string window; 
        window.reserve(128 * 1024);
        char buffer[32768];
        int bytes_read;
        
        logging::Logger::info("Parsing reference dataset...");
        
        std::vector<std::pair<std::array<float, 14>, bool>> raw_data;
        raw_data.reserve(3000000);

        while ((bytes_read = gzread(file, buffer, sizeof(buffer))) > 0) {
            window.append(buffer, bytes_read);
            size_t start = 0;
            while (true) {
                size_t obj_start = window.find("{", start);
                if (obj_start == std::string::npos) break;
                size_t obj_end = window.find("}", obj_start);
                if (obj_end == std::string::npos) break; 
                
                std::string_view obj = std::string_view(window).substr(obj_start, obj_end - obj_start + 1);
                std::vector<float> v = FastParser::extract_vector(obj);
                
                if (v.size() == 14) {
                    std::array<float, 14> arr;
                    std::copy(v.begin(), v.end(), arr.begin());
                    std::string_view label = FastParser::find_string(obj, "label");
                    raw_data.push_back({arr, label == "fraud"});
                }
                start = obj_end + 1;
            }
            if (start > 0) window.erase(0, start);
        }
        gzclose(file);

        matcher->train_and_build(raw_data);
    }
};

} // namespace vector_search
} // namespace adapters
} // namespace infrastructure

#endif
