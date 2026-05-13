#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <string_view>
#include <string>
#include <vector>
#include <charconv>
#include <algorithm>

namespace infrastructure {
namespace parser {

class FastJsonScanner {
public:
    static constexpr std::string_view K_ID = "\"id\"";
    static constexpr std::string_view K_TX = "\"transaction\"";
    static constexpr std::string_view K_CUST = "\"customer\"";
    static constexpr std::string_view K_MERCH = "\"merchant\"";
    static constexpr std::string_view K_TERM = "\"terminal\"";
    static constexpr std::string_view K_LAST = "\"last_transaction\"";
    static constexpr std::string_view K_AMOUNT = "\"amount\"";
    static constexpr std::string_view K_AVG = "\"avg_amount\"";
    static constexpr std::string_view K_INST = "\"installments\"";
    static constexpr std::string_view K_REQ = "\"requested_at\"";
    static constexpr std::string_view K_COUNT = "\"tx_count_24h\"";
    static constexpr std::string_view K_MCC = "\"mcc\"";
    static constexpr std::string_view K_ONLINE = "\"is_online\"";
    static constexpr std::string_view K_CARD = "\"card_present\"";
    static constexpr std::string_view K_HOME = "\"km_from_home\"";
    static constexpr std::string_view K_TS = "\"timestamp\"";
    static constexpr std::string_view K_KM = "\"km_from_current\"";
    static constexpr std::string_view K_LABEL = "\"label\"";

    static std::string_view find_string(std::string_view json, std::string_view search_key) {
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string_view::npos) return "";
        
        size_t colon_pos = json.find(":", key_pos + search_key.size());
        if (colon_pos == std::string_view::npos) return "";
        
        size_t start = json.find("\"", colon_pos + 1);
        if (start == std::string_view::npos) return "";
        
        size_t end = json.find("\"", start + 1);
        if (end == std::string_view::npos) return "";
        
        return json.substr(start + 1, end - start - 1);
    }

    static double find_double(std::string_view json, std::string_view search_key) {
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string_view::npos) return 0.0;
        
        size_t colon_pos = json.find(":", key_pos + search_key.size());
        if (colon_pos == std::string_view::npos) return 0.0;
        
        size_t start = json.find_first_of("0123456789-", colon_pos + 1);
        if (start == std::string_view::npos) return 0.0;
        
        size_t end = json.find_first_not_of("0123456789.eE-", start);
        std::string_view val_str = json.substr(start, (end == std::string_view::npos) ? json.size() - start : end - start);
        
        double result = 0.0;
#if __cpp_lib_to_chars >= 201611L || defined(_MSC_VER)
        auto [ptr, ec] = std::from_chars(val_str.data(), val_str.data() + val_str.size(), result);
        if (ec == std::errc()) return result;
#endif
        // Fallback via strtod para compiladores sem from_chars<double>
        char buf[32];
        size_t len = std::min(val_str.size(), sizeof(buf) - 1);
        std::copy(val_str.begin(), val_str.begin() + len, buf);
        buf[len] = '\0';
        return std::strtod(buf, nullptr);
    }

    static int find_int(std::string_view json, std::string_view search_key) {
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string_view::npos) return 0;

        size_t colon_pos = json.find(":", key_pos + search_key.size());
        if (colon_pos == std::string_view::npos) return 0;

        size_t start = json.find_first_of("0123456789-", colon_pos + 1);
        if (start == std::string_view::npos) return 0;

        int result = 0;
        std::from_chars(json.data() + start, json.data() + json.size(), result);
        return result;
    }

    static bool find_bool(std::string_view json, std::string_view search_key) {
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string_view::npos) return false;

        size_t colon_pos = json.find(":", key_pos + search_key.size());
        if (colon_pos == std::string_view::npos) return false;

        size_t val_start = json.find_first_not_of(" \t\n\r", colon_pos + 1);
        if (val_start == std::string_view::npos || val_start + 4 > json.size()) return false;

        return (json[val_start]     == 't' &&
                json[val_start + 1] == 'r' &&
                json[val_start + 2] == 'u' &&
                json[val_start + 3] == 'e');
    }

    static std::string_view find_object(std::string_view json, std::string_view search_key) {
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string_view::npos) return "";
        
        size_t colon_pos = json.find(":", key_pos + search_key.size());
        if (colon_pos != std::string_view::npos) {
            size_t next_char = json.find_first_not_of(" \t\n\r", colon_pos + 1);
            if (next_char != std::string_view::npos && next_char + 4 <= json.size() && json.substr(next_char, 4) == "null") {
                return "";
            }
        }

        size_t start = json.find("{", key_pos + search_key.size());
        if (start == std::string_view::npos) return "";
        
        int level = 0;
        for (size_t i = start; i < json.size(); ++i) {
            if (json[i] == '{') level++;
            else if (json[i] == '}') {
                level--;
                if (level == 0) return json.substr(start, i - start + 1);
            }
        }
        return "";
    }

    static std::vector<float> extract_vector(std::string_view object) {
        std::vector<float> v;
        v.reserve(14);
        
        size_t start = object.find("[");
        if (start == std::string_view::npos) return v;
        
        size_t current = start + 1;
        char buf[64];
        
        while (v.size() < 14) {
            size_t num_start = object.find_first_of("0123456789.-", current);
            if (num_start == std::string_view::npos) break;
            
            size_t num_end = object.find_first_of(",]", num_start);
            size_t len = num_end - num_start;
            if (len >= 64) break;
            
            std::copy(object.begin() + num_start, object.begin() + num_end, buf);
            buf[len] = '\0';
            
            v.push_back(std::strtof(buf, nullptr));
            
            if (object[num_end] == ']') break;
            current = num_end + 1;
        }
        return v;
    }
};

} // namespace parser
} // namespace infrastructure

#endif
