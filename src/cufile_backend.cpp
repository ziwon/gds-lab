#include "backends.hpp"

#include <cuda_runtime.h>
#include <cufile.h>

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

void cufile_check(CUfileError_t status, const char* op) {
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error(std::string("SKIPPED: ") + op + " failed with cuFile error " +
                                 std::to_string(static_cast<int>(status.err)));
    }
}

bool env_true(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0;
}
}  // namespace

BenchmarkResult run_cufile(const std::string& path, std::size_t bytes) {
    cufile_check(cuFileDriverOpen(), "cuFileDriverOpen");

    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        cuFileDriverClose();
        throw std::runtime_error(std::string("SKIPPED: open(O_DIRECT): ") + std::strerror(errno));
    }

    CUfileDescr_t descriptor{};
    descriptor.handle.fd = fd;
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t handle{};

    CUfileError_t status = cuFileHandleRegister(&handle, &descriptor);
    if (status.err != CU_FILE_SUCCESS) {
        ::close(fd);
        cuFileDriverClose();
        cufile_check(status, "cuFileHandleRegister");
    }

    void* device = nullptr;
    cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");

    const auto start = std::chrono::steady_clock::now();
    const ssize_t n = cuFileRead(handle, device, bytes, 0, 0);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto end = std::chrono::steady_clock::now();

    cudaFree(device);
    cuFileHandleDeregister(handle);
    ::close(fd);
    cuFileDriverClose();

    if (n < 0) {
        throw std::runtime_error("SKIPPED: cuFileRead failed with code " + std::to_string(n));
    }
    if (static_cast<std::size_t>(n) != bytes) {
        throw std::runtime_error("cuFileRead returned fewer bytes than requested");
    }

    const double seconds = std::chrono::duration<double>(end - start).count();
    const char* label = env_true("CUFILE_FORCE_COMPAT_MODE") ? "cufile-forced-compat" : "cufile-auto-unverified-path";
    return {label, bytes, seconds};
}
