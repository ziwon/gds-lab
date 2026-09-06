#include "backends.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
void cuda_check(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

__global__ void touch_kernel(const unsigned char* data, std::size_t n, unsigned long long* checksum) {
    __shared__ unsigned long long partial[256];
    const unsigned int tid = threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    unsigned long long local = 0;

    while (i < n) {
        local += data[i];
        i += stride;
    }
    partial[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) partial[tid] += partial[tid + offset];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(checksum, partial[0]);
}

std::size_t pread_exact(int fd, void* buffer, std::size_t bytes, off_t offset) {
    auto* out = static_cast<unsigned char*>(buffer);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(fd, out + done, bytes - done, offset + static_cast<off_t>(done));
        if (n < 0) throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
        if (n == 0) throw std::runtime_error("unexpected EOF: test file is smaller than --bytes");
        done += static_cast<std::size_t>(n);
    }
    return done;
}
}  // namespace

BenchmarkResult run_overlap(const std::string& path, std::size_t bytes, std::size_t chunk_bytes) {
    if (chunk_bytes == 0) throw std::invalid_argument("--chunk-bytes must be > 0");

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open(" + path + "): " + std::strerror(errno));

    void* host[2] = {nullptr, nullptr};
    unsigned char* device[2] = {nullptr, nullptr};
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaEvent_t copy_done[2] = {nullptr, nullptr};
    cudaEvent_t compute_done[2] = {nullptr, nullptr};
    unsigned long long* device_checksum = nullptr;

    for (int slot = 0; slot < 2; ++slot) {
        cuda_check(cudaHostAlloc(&host[slot], chunk_bytes, cudaHostAllocDefault), "cudaHostAlloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device[slot]), chunk_bytes), "cudaMalloc");
        cuda_check(cudaEventCreateWithFlags(&copy_done[slot], cudaEventDisableTiming), "cudaEventCreate");
        cuda_check(cudaEventCreateWithFlags(&compute_done[slot], cudaEventDisableTiming), "cudaEventCreate");
    }
    cuda_check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "cudaStreamCreate(copy)");
    cuda_check(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), "cudaStreamCreate(compute)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_checksum), sizeof(unsigned long long)), "cudaMalloc(checksum)");
    cuda_check(cudaMemset(device_checksum, 0, sizeof(unsigned long long)), "cudaMemset(checksum)");

    const auto start = std::chrono::steady_clock::now();
    std::size_t processed = 0;
    std::size_t step = 0;

    while (processed < bytes) {
        const int slot = static_cast<int>(step & 1U);
        if (step >= 2) cuda_check(cudaEventSynchronize(compute_done[slot]), "cudaEventSynchronize(reuse)");

        const std::size_t current = std::min(chunk_bytes, bytes - processed);
        pread_exact(fd, host[slot], current, static_cast<off_t>(processed));

        cuda_check(cudaMemcpyAsync(device[slot], host[slot], current, cudaMemcpyHostToDevice, copy_stream),
                   "cudaMemcpyAsync");
        cuda_check(cudaEventRecord(copy_done[slot], copy_stream), "cudaEventRecord(copy)");
        cuda_check(cudaStreamWaitEvent(compute_stream, copy_done[slot], 0), "cudaStreamWaitEvent");

        constexpr int threads = 256;
        const int blocks = static_cast<int>(std::min<std::size_t>(4096, (current + threads - 1) / threads));
        touch_kernel<<<blocks, threads, 0, compute_stream>>>(device[slot], current, device_checksum);
        cuda_check(cudaGetLastError(), "touch_kernel launch");
        cuda_check(cudaEventRecord(compute_done[slot], compute_stream), "cudaEventRecord(compute)");

        processed += current;
        ++step;
    }

    cuda_check(cudaStreamSynchronize(compute_stream), "cudaStreamSynchronize(compute)");
    const auto end = std::chrono::steady_clock::now();

    unsigned long long checksum = 0;
    cuda_check(cudaMemcpy(&checksum, device_checksum, sizeof(checksum), cudaMemcpyDeviceToHost), "cudaMemcpy(checksum)");

    cudaFree(device_checksum);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    for (int slot = 0; slot < 2; ++slot) {
        cudaEventDestroy(copy_done[slot]);
        cudaEventDestroy(compute_done[slot]);
        cudaFree(device[slot]);
        cudaFreeHost(host[slot]);
    }
    ::close(fd);

    const double seconds = std::chrono::duration<double>(end - start).count();
    BenchmarkResult result{"double-buffer-overlap", processed, seconds};
    result.checksum = checksum;
    result.checksum_valid = true;
    return result;
}
