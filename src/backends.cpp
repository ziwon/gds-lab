#include "backends.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef GDSLAB_HAS_CUFILE
#define GDSLAB_HAS_CUFILE 0
#endif

namespace {
constexpr std::size_t kDirectAlignment = 4096;
constexpr std::size_t kReadChunk = 16ULL * 1024ULL * 1024ULL;

void cuda_check(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

int open_readonly(const std::string& path, bool direct) {
    const int flags = direct ? (O_RDONLY | O_DIRECT) : O_RDONLY;
    const int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        throw std::runtime_error("open(" + path + "): " + std::strerror(errno));
    }
    return fd;
}

void read_exact(int fd, void* buffer, std::size_t bytes, bool direct) {
    auto* out = static_cast<unsigned char*>(buffer);
    std::size_t done = 0;

    while (done < bytes) {
        const std::size_t wanted = std::min(kReadChunk, bytes - done);
        if (direct && ((wanted % kDirectAlignment) != 0 || (done % kDirectAlignment) != 0)) {
            throw std::runtime_error("O_DIRECT read size/offset must be 4 KiB aligned");
        }

        const ssize_t n = ::pread(fd, out + done, wanted, static_cast<off_t>(done));
        if (n < 0) {
            throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error("unexpected EOF: test file is smaller than --bytes");
        }
        if (direct && (static_cast<std::size_t>(n) % kDirectAlignment) != 0) {
            throw std::runtime_error("O_DIRECT returned a non-aligned short read");
        }
        done += static_cast<std::size_t>(n);
    }
}

double elapsed_seconds(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}
}  // namespace

bool cufile_compiled() {
    return GDSLAB_HAS_CUFILE != 0;
}

BenchmarkResult run_pageable(const std::string& path, std::size_t bytes) {
    const int fd = open_readonly(path, false);
    std::vector<unsigned char> host(bytes);
    void* device = nullptr;
    cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");

    const auto start = std::chrono::steady_clock::now();
    read_exact(fd, host.data(), bytes, false);
    cuda_check(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    const auto end = std::chrono::steady_clock::now();

    cudaFree(device);
    ::close(fd);
    return {"pageable-sync", bytes, elapsed_seconds(start, end)};
}

BenchmarkResult run_pinned(const std::string& path, std::size_t bytes) {
    const int fd = open_readonly(path, false);
    void* host = nullptr;
    void* device = nullptr;
    cudaStream_t stream = nullptr;

    cuda_check(cudaHostAlloc(&host, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    const auto start = std::chrono::steady_clock::now();
    read_exact(fd, host, bytes, false);
    cuda_check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto end = std::chrono::steady_clock::now();

    cudaStreamDestroy(stream);
    cudaFree(device);
    cudaFreeHost(host);
    ::close(fd);
    return {"pinned-async", bytes, elapsed_seconds(start, end)};
}

BenchmarkResult run_direct(const std::string& path, std::size_t bytes) {
    if ((bytes % kDirectAlignment) != 0) {
        throw std::runtime_error("SKIPPED: --bytes must be a multiple of 4096 for O_DIRECT");
    }

    const int fd = open_readonly(path, true);
    void* host = nullptr;
    if (::posix_memalign(&host, kDirectAlignment, bytes) != 0) {
        ::close(fd);
        throw std::runtime_error("posix_memalign failed");
    }

    void* device = nullptr;
    cudaStream_t stream = nullptr;
    cuda_check(cudaHostRegister(host, bytes, cudaHostRegisterDefault), "cudaHostRegister");
    cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    const auto start = std::chrono::steady_clock::now();
    read_exact(fd, host, bytes, true);
    cuda_check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto end = std::chrono::steady_clock::now();

    cudaStreamDestroy(stream);
    cudaFree(device);
    cudaHostUnregister(host);
    std::free(host);
    ::close(fd);
    return {"odirect-pinned-async", bytes, elapsed_seconds(start, end)};
}

#if !GDSLAB_HAS_CUFILE
BenchmarkResult run_cufile(const std::string&, std::size_t) {
    throw std::runtime_error("SKIPPED: cuFile backend was not built (cufile.h/libcufile not found)");
}
#endif
