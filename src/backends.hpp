#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// One knob set for every backend. Experiments vary exactly one of these at a
// time, so they live together rather than as per-function parameters.
struct BenchmarkOptions {
    std::string path;
    std::size_t bytes = 0;
    // Storage read granularity. Experiment 04 sweeps this; it is independent
    // of chunk_bytes, which is the overlap pipeline's buffer size.
    std::size_t read_chunk = 16ULL * 1024ULL * 1024ULL;
    std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    // Byte offset of the region to read. Iterations advance this so a repeat
    // run touches new file extents instead of re-reading the page cache.
    std::size_t offset = 0;
    // Experiment 05: visit read_chunk-sized blocks in shuffled order. The
    // destination stays position-correct, so the checksum must not change.
    bool random_access = false;
    std::uint64_t seed = 0;
    // Open with O_DIRECT where the backend supports both modes (overlap).
    bool direct = false;
};

struct BenchmarkResult {
    std::string label;
    std::size_t bytes = 0;
    double seconds = 0.0;
    // Storage read and H2D copy measured separately. Experiments 01 and 02 ask
    // which of the two a change actually moved, which a single total cannot
    // answer. Pipelined backends overlap them, so they report split_valid=false.
    double read_seconds = 0.0;
    double copy_seconds = 0.0;
    bool split_valid = false;
    // Sum of every byte in the region, computed on the GPU after the timed
    // section. Identical across backends, access patterns and worker counts
    // when the transfer is correct.
    std::uint64_t checksum = 0;
    bool checksum_valid = false;
};

BenchmarkResult run_pageable(const BenchmarkOptions& options);
BenchmarkResult run_pinned(const BenchmarkOptions& options);
BenchmarkResult run_direct(const BenchmarkOptions& options);
BenchmarkResult run_overlap(const BenchmarkOptions& options);
BenchmarkResult run_cufile(const BenchmarkOptions& options);

// Sums bytes of a device buffer. Shared so every backend reports a checksum
// computed the same way.
std::uint64_t device_checksum(const void* device, std::size_t bytes);

bool cufile_compiled();
