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
        << "  --bytes SIZE        bytes to process per iteration (default: 1G)\n"
        << "  --read-chunk SIZE   storage read granularity (default: 16M, experiment 04)\n"
        << "  --chunk-bytes SIZE  overlap pipeline buffer size (default: 64M)\n"
        << "  --offset SIZE       first byte of the region to read (default: 0)\n"
        << "  --iterations N      repeat count (default: 1)\n"
        << "  --fixed-offset      re-read the same region every iteration\n"
        << "                      (default: advance by --bytes, so repeats are not cache hits)\n"
        << "  --random-access     visit read blocks in shuffled order (experiment 05)\n"
        << "  --seed N            shuffle seed (default: 0)\n"
        << "  --direct            use O_DIRECT for the overlap backend\n"
        << "  --help              show this help\n";
}

bool env_true(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

BenchmarkResult dispatch(const std::string& backend, const BenchmarkOptions& options) {
    if (backend == "pageable") return run_pageable(options);
    if (backend == "pinned") return run_pinned(options);
    if (backend == "direct") return run_direct(options);
    if (backend == "overlap") return run_overlap(options);
    if (backend == "cufile") return run_cufile(options);
    throw std::invalid_argument("unknown backend: " + backend);
}
}  // namespace

int main(int argc, char** argv) {
    std::string backend;
    BenchmarkOptions options;
    options.bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    int iterations = 1;
    bool fixed_offset = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
                return argv[++i];
            };

            if (arg == "--backend") backend = require_value("--backend");
            else if (arg == "--file") options.path = require_value("--file");
            else if (arg == "--bytes") options.bytes = parse_bytes(require_value("--bytes"));
            else if (arg == "--read-chunk") options.read_chunk = parse_bytes(require_value("--read-chunk"));
            else if (arg == "--chunk-bytes") options.chunk_bytes = parse_bytes(require_value("--chunk-bytes"));
            else if (arg == "--offset") options.offset = parse_bytes(require_value("--offset"));
            else if (arg == "--iterations") iterations = std::stoi(require_value("--iterations"));
            else if (arg == "--fixed-offset") fixed_offset = true;
            else if (arg == "--random-access") options.random_access = true;
            else if (arg == "--seed") options.seed = std::stoull(require_value("--seed"));
            else if (arg == "--direct") options.direct = true;
            else if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
            else throw std::invalid_argument("unknown argument: " + arg);
        }

        if (backend.empty() || options.path.empty()) {
            usage(argv[0]);
            return 2;
        }
        if (iterations < 1) throw std::invalid_argument("--iterations must be >= 1");
        if (options.bytes == 0) throw std::invalid_argument("--bytes must be > 0");
        if (options.read_chunk == 0) throw std::invalid_argument("--read-chunk must be > 0");
        if (options.chunk_bytes == 0) throw std::invalid_argument("--chunk-bytes must be > 0");

        std::cout << "cufile_compiled=" << (cufile_compiled() ? "yes" : "no") << '\n';
        if (backend == "cufile") {
            std::cout << "cufile_requested_mode="
                      << (env_true("CUFILE_FORCE_COMPAT_MODE") ? "forced-compat" : "auto") << '\n';
        }
        if (!fixed_offset && iterations > 1) {
            std::cout << "offset_policy=advance (each iteration reads the next "
                      << options.bytes << " bytes)\n";
        } else if (fixed_offset && iterations > 1) {
            std::cout << "offset_policy=fixed (iterations after the first are page-cache hits)\n";
        }

        const std::size_t base_offset = options.offset;
        for (int iteration = 1; iteration <= iterations; ++iteration) {
            BenchmarkOptions current = options;
            if (!fixed_offset) {
                current.offset = base_offset + static_cast<std::size_t>(iteration - 1) * options.bytes;
            }

            const BenchmarkResult result = dispatch(backend, current);

            const double gbps = static_cast<double>(result.bytes) / result.seconds / 1.0e9;
            std::cout << std::fixed << std::setprecision(4)
                      << "iteration=" << iteration
                      << " backend=" << result.label
                      << " offset=" << current.offset
                      << " bytes=" << result.bytes
                      << " seconds=" << result.seconds
                      << " effective_GBps=" << gbps;
            if (result.split_valid) {
                const double read_gbps =
                    result.read_seconds > 0.0
                        ? static_cast<double>(result.bytes) / result.read_seconds / 1.0e9
                        : 0.0;
                const double copy_gbps =
                    result.copy_seconds > 0.0
                        ? static_cast<double>(result.bytes) / result.copy_seconds / 1.0e9
                        : 0.0;
                std::cout << " read_seconds=" << result.read_seconds
                          << " copy_seconds=" << result.copy_seconds
                          << " read_GBps=" << read_gbps
                          << " h2d_GBps=" << copy_gbps;
            }
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
