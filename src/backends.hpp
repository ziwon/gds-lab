#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// One knob set for every backend. Experiments vary exactly one of these at a
// time, so they live together rather than as per-function parameters.
struct BenchmarkOptions {
    std::string path;
    std::size_t bytes = 0;
    std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
};

struct BenchmarkResult {
    std::string label;
    std::size_t bytes = 0;
    double seconds = 0.0;
    std::uint64_t checksum = 0;
    bool checksum_valid = false;
};

BenchmarkResult run_pageable(const BenchmarkOptions& options);
BenchmarkResult run_pinned(const BenchmarkOptions& options);
BenchmarkResult run_direct(const BenchmarkOptions& options);
BenchmarkResult run_overlap(const BenchmarkOptions& options);
BenchmarkResult run_cufile(const BenchmarkOptions& options);

bool cufile_compiled();
