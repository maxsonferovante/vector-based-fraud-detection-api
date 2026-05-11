#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>

namespace infrastructure {
namespace logging {

class Logger {
public:
    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    static void info(const std::string& message) {
        std::cout << "[" << get_timestamp() << "] [INFO] " << message << std::endl;
    }

    static void error(const std::string& message) {
        std::cerr << "[" << get_timestamp() << "] [ERROR] " << message << std::endl;
    }
};

} // namespace logging
} // namespace infrastructure

#endif
