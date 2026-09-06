#include "backends.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
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
// Conservative: 512 is this NVMe's logical_block_size, but ext4/xfs on some
// configurations still require 4 KiB for O_DIRECT. inspect.sh reports the
// device value so a deliberate override can be justified from evidence.
constexpr std::size_t kDirectAlignment = 4096;
constexpr std::size_t kReadChunk = 16ULL * 1024ULL * 1024ULL;

void cuda_check(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

// Small RAII guards. Every backend throws on error, and the throw sites sit
// between acquisitions, so releasing by hand would leak on the failure path.
struct FdGuard {
    int fd = -1;
    explicit FdGuard(int value) : fd(value) {}
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
    void* ptr = nullptr;
    explicit PinnedBuffer(std::size_t bytes) {
        cuda_check(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    }
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

// posix_memalign'd host memory page-locked after the fact, which is what an
// application doing its own O_DIRECT buffer management would do.
struct RegisteredBuffer {
    void* ptr = nullptr;
    bool registered = false;
    RegisteredBuffer(std::size_t bytes, std::size_t alignment) {
        if (::posix_memalign(&ptr, alignment, bytes) != 0) {
            ptr = nullptr;
            throw std::runtime_error("posix_memalign failed");
        }
        cuda_check(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault), "cudaHostRegister");
        registered = true;
    }
    ~RegisteredBuffer() {
        if (registered) cudaHostUnregister(ptr);
        std::free(ptr);
    }
    RegisteredBuffer(const RegisteredBuffer&) = delete;
    RegisteredBuffer& operator=(const RegisteredBuffer&) = delete;
};

struct StreamGuard {
    cudaStream_t stream = nullptr;
    StreamGuard() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~StreamGuard() { if (stream) cudaStreamDestroy(stream); }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
};

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

BenchmarkResult run_pageable(const BenchmarkOptions& options) {
    FdGuard fd(open_readonly(options.path, false));
    std::vector<unsigned char> host(options.bytes);
    DeviceBuffer device(options.bytes);

    const auto start = std::chrono::steady_clock::now();
    read_exact(fd.fd, host.data(), options.bytes, false);
    cuda_check(cudaMemcpy(device.ptr, host.data(), options.bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy");
    const auto end = std::chrono::steady_clock::now();

    return {"pageable-sync", options.bytes, elapsed_seconds(start, end)};
}

BenchmarkResult run_pinned(const BenchmarkOptions& options) {
    FdGuard fd(open_readonly(options.path, false));
    PinnedBuffer host(options.bytes);
    DeviceBuffer device(options.bytes);
    StreamGuard stream;

    const auto start = std::chrono::steady_clock::now();
    read_exact(fd.fd, host.ptr, options.bytes, false);
    cuda_check(cudaMemcpyAsync(device.ptr, host.ptr, options.bytes, cudaMemcpyHostToDevice,
                               stream.stream), "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize");
    const auto end = std::chrono::steady_clock::now();

    return {"pinned-async", options.bytes, elapsed_seconds(start, end)};
}

BenchmarkResult run_direct(const BenchmarkOptions& options) {
    if ((options.bytes % kDirectAlignment) != 0) {
        throw std::runtime_error("SKIPPED: --bytes must be a multiple of 4096 for O_DIRECT");
    }

    FdGuard fd(open_readonly(options.path, true));
    RegisteredBuffer host(options.bytes, kDirectAlignment);
    DeviceBuffer device(options.bytes);
    StreamGuard stream;

    const auto start = std::chrono::steady_clock::now();
    read_exact(fd.fd, host.ptr, options.bytes, true);
    cuda_check(cudaMemcpyAsync(device.ptr, host.ptr, options.bytes, cudaMemcpyHostToDevice,
                               stream.stream), "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize");
    const auto end = std::chrono::steady_clock::now();

    return {"odirect-pinned-async", options.bytes, elapsed_seconds(start, end)};
}

#if !GDSLAB_HAS_CUFILE
BenchmarkResult run_cufile(const BenchmarkOptions&) {
    throw std::runtime_error("SKIPPED: cuFile backend was not built (cufile.h/libcufile not found)");
}
#endif
