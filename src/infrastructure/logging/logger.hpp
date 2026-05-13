#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include <array>

namespace infrastructure {
namespace logging {

class Logger {
public:
    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::array<char, 20> buf;
        struct tm tm_buf{};
        localtime_r(&t, &tm_buf); // POSIX thread-safe; std::localtime usa buffer estático global
        std::strftime(buf.data(), buf.size(), "%Y-%m-%d %H:%M:%S", &tm_buf);
        return std::string(buf.data());
    }

    static void info(const std::string& message) {
        std::cout << '[' << get_timestamp() << "] [INFO] " << message << '\n';
    }

    static void error(const std::string& message) {
        std::cerr << '[' << get_timestamp() << "] [ERROR] " << message << '\n';
        std::cerr.flush();
    }
};

} // namespace logging
} // namespace infrastructure

#endif
