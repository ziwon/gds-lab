#include "backends.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::size_t parse_bytes(std::string value) {
    if (value.empty()) {
        throw std::invalid_argument("empty size");
    }

    std::size_t multiplier = 1;
    const char suffix =
        static_cast<char>(std::toupper(static_cast<unsigned char>(value.back())));
    if (!std::isdigit(static_cast<unsigned char>(suffix))) {
        value.pop_back();
        switch (suffix) {
            case 'K': multiplier = 1024ULL; break;
            case 'M': multiplier = 1024ULL * 1024ULL; break;
            case 'G': multiplier = 1024ULL * 1024ULL * 1024ULL; break;
            default: throw std::invalid_argument("size suffix must be K, M, or G");
        }
    }
    if (value.empty()) {
        throw std::invalid_argument("size needs a number before the suffix");
    }

    const unsigned long long scalar = std::stoull(value);
    if (scalar > std::numeric_limits<std::size_t>::max() / multiplier) {
        throw std::invalid_argument("size overflows size_t: " + value);
    }
    return static_cast<std::size_t>(scalar) * multiplier;
}

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --backend NAME --file PATH [options]\n\n"
        << "Backends:\n"
        << "  pageable   buffered POSIX -> pageable host RAM -> synchronous H2D\n"
        << "  pinned     buffered POSIX -> pinned host RAM -> asynchronous H2D\n"
        << "  direct     O_DIRECT -> aligned pinned host RAM -> asynchronous H2D\n"
        << "  overlap    double-buffered read/H2D/GPU-touch pipeline\n"
        << "  cufile     cuFile API; active direct/compat path must be verified externally\n\n"
        << "Options:\n"
        << "  --bytes SIZE        bytes to process (default: 1G)\n"
        << "  --chunk-bytes SIZE  overlap chunk size (default: 64M)\n"
        << "  --iterations N      repeat count (default: 1)\n"
        << "  --help              show this help\n";
}

bool env_true(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}
}  // namespace

int main(int argc, char** argv) {
    std::string backend;
    std::string file;
    std::size_t bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    int iterations = 1;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
                return argv[++i];
            };

            if (arg == "--backend") backend = require_value("--backend");
            else if (arg == "--file") file = require_value("--file");
            else if (arg == "--bytes") bytes = parse_bytes(require_value("--bytes"));
            else if (arg == "--chunk-bytes") chunk_bytes = parse_bytes(require_value("--chunk-bytes"));
            else if (arg == "--iterations") iterations = std::stoi(require_value("--iterations"));
            else if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
            else throw std::invalid_argument("unknown argument: " + arg);
        }

        if (backend.empty() || file.empty()) {
            usage(argv[0]);
            return 2;
        }
        if (iterations < 1) throw std::invalid_argument("--iterations must be >= 1");
        if (bytes == 0) throw std::invalid_argument("--bytes must be > 0");
        if (chunk_bytes == 0) throw std::invalid_argument("--chunk-bytes must be > 0");

        std::cout << "cufile_compiled=" << (cufile_compiled() ? "yes" : "no") << '\n';
        if (backend == "cufile") {
            std::cout << "cufile_requested_mode="
                      << (env_true("CUFILE_FORCE_COMPAT_MODE") ? "forced-compat" : "auto") << '\n';
        }

        for (int iteration = 1; iteration <= iterations; ++iteration) {
            BenchmarkResult result;
            if (backend == "pageable") result = run_pageable(file, bytes);
            else if (backend == "pinned") result = run_pinned(file, bytes);
            else if (backend == "direct") result = run_direct(file, bytes);
            else if (backend == "overlap") result = run_overlap(file, bytes, chunk_bytes);
            else if (backend == "cufile") result = run_cufile(file, bytes);
            else throw std::invalid_argument("unknown backend: " + backend);

            const double gbps = static_cast<double>(result.bytes) / result.seconds / 1.0e9;
            std::cout << std::fixed << std::setprecision(4)
                      << "iteration=" << iteration
                      << " backend=" << result.label
                      << " bytes=" << result.bytes
                      << " seconds=" << result.seconds
                      << " effective_GBps=" << gbps;
            if (result.checksum_valid) std::cout << " checksum=" << result.checksum;
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        const std::string message = ex.what();
        if (message.rfind("SKIPPED:", 0) == 0) {
            std::cerr << message << '\n';
            return 3;
        }
        std::cerr << "ERROR: " << message << '\n';
        return 1;
    }
}
