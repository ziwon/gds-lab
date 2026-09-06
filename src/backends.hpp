#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct BenchmarkResult {
    std::string label;
    std::size_t bytes = 0;
    double seconds = 0.0;
    std::uint64_t checksum = 0;
    bool checksum_valid = false;
};

BenchmarkResult run_pageable(const std::string& path, std::size_t bytes);
BenchmarkResult run_pinned(const std::string& path, std::size_t bytes);
BenchmarkResult run_direct(const std::string& path, std::size_t bytes);
BenchmarkResult run_overlap(const std::string& path, std::size_t bytes, std::size_t chunk_bytes);
BenchmarkResult run_cufile(const std::string& path, std::size_t bytes);

bool cufile_compiled();
