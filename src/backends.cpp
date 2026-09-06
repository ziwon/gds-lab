#include "backends.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <random>
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

void pread_all(int fd, unsigned char* out, std::size_t bytes, off_t offset, bool direct) {
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(fd, out + done, bytes - done, offset + static_cast<off_t>(done));
        if (n < 0) {
            throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error(
                "unexpected EOF: file is smaller than --offset + --bytes");
        }
        if (direct && (static_cast<std::size_t>(n) % kDirectAlignment) != 0) {
            throw std::runtime_error("O_DIRECT returned a non-aligned short read");
        }
        done += static_cast<std::size_t>(n);
    }
}

// Reads [offset, offset + bytes) into buffer in read_chunk-sized blocks. With
// random_access the block visit order is shuffled, but each block still lands
// at its own position in the buffer, so the checksum is order-independent and
// can prove the randomized path read the same data.
void read_region(int fd, void* buffer, const BenchmarkOptions& options, bool direct) {
    auto* out = static_cast<unsigned char*>(buffer);
    const std::size_t chunk = options.read_chunk;
    if (chunk == 0) throw std::invalid_argument("--read-chunk must be > 0");

    if (direct) {
        if ((options.offset % kDirectAlignment) != 0 || (chunk % kDirectAlignment) != 0) {
            throw std::runtime_error(
                "SKIPPED: --offset and --read-chunk must be multiples of 4096 for O_DIRECT");
        }
    }

    const std::size_t blocks = (options.bytes + chunk - 1) / chunk;
    std::vector<std::size_t> order(blocks);
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (options.random_access) {
        std::mt19937_64 rng(options.seed);
        std::shuffle(order.begin(), order.end(), rng);
    }

    for (const std::size_t block : order) {
        const std::size_t block_offset = block * chunk;
        const std::size_t wanted = std::min(chunk, options.bytes - block_offset);
        pread_all(fd, out + block_offset, wanted,
                  static_cast<off_t>(options.offset + block_offset), direct);
    }
}

double seconds_between(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

std::string pattern_suffix(const BenchmarkOptions& options) {
    return options.random_access ? "-random" : "";
}

BenchmarkResult finish(const std::string& label, const BenchmarkOptions& options,
                       std::chrono::steady_clock::time_point t0,
                       std::chrono::steady_clock::time_point t1,
                       std::chrono::steady_clock::time_point t2,
                       const void* device) {
    BenchmarkResult result;
    result.label = label + pattern_suffix(options);
    result.bytes = options.bytes;
    result.seconds = seconds_between(t0, t2);
    result.read_seconds = seconds_between(t0, t1);
    result.copy_seconds = seconds_between(t1, t2);
    result.split_valid = true;
    result.checksum = device_checksum(device, options.bytes);
    result.checksum_valid = true;
    return result;
}
}  // namespace

bool cufile_compiled() {
    return GDSLAB_HAS_CUFILE != 0;
}

BenchmarkResult run_pageable(const BenchmarkOptions& options) {
    FdGuard fd(open_readonly(options.path, false));
    // Touched by the allocator before the timer, so this measures the read and
    // the copy, not first-touch page faults on the destination.
    std::vector<unsigned char> host(options.bytes);
    DeviceBuffer device(options.bytes);

    const auto t0 = std::chrono::steady_clock::now();
    read_region(fd.fd, host.data(), options, false);
    const auto t1 = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(device.ptr, host.data(), options.bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy");
    const auto t2 = std::chrono::steady_clock::now();

    return finish("pageable-sync", options, t0, t1, t2, device.ptr);
}

BenchmarkResult run_pinned(const BenchmarkOptions& options) {
    FdGuard fd(open_readonly(options.path, false));
    PinnedBuffer host(options.bytes);
    DeviceBuffer device(options.bytes);
    StreamGuard stream;

    const auto t0 = std::chrono::steady_clock::now();
    read_region(fd.fd, host.ptr, options, false);
    const auto t1 = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpyAsync(device.ptr, host.ptr, options.bytes, cudaMemcpyHostToDevice,
                               stream.stream), "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize");
    const auto t2 = std::chrono::steady_clock::now();

    return finish("pinned-async", options, t0, t1, t2, device.ptr);
}

BenchmarkResult run_direct(const BenchmarkOptions& options) {
    if ((options.bytes % kDirectAlignment) != 0) {
        throw std::runtime_error("SKIPPED: --bytes must be a multiple of 4096 for O_DIRECT");
    }

    FdGuard fd(open_readonly(options.path, true));
    RegisteredBuffer host(options.bytes, kDirectAlignment);
    DeviceBuffer device(options.bytes);
    StreamGuard stream;

    const auto t0 = std::chrono::steady_clock::now();
    read_region(fd.fd, host.ptr, options, true);
    const auto t1 = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpyAsync(device.ptr, host.ptr, options.bytes, cudaMemcpyHostToDevice,
                               stream.stream), "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize");
    const auto t2 = std::chrono::steady_clock::now();

    return finish("odirect-pinned-async", options, t0, t1, t2, device.ptr);
}

#if !GDSLAB_HAS_CUFILE
BenchmarkResult run_cufile(const BenchmarkOptions&) {
    throw std::runtime_error("SKIPPED: cuFile backend was not built (cufile.h/libcufile not found)");
}
#endif
