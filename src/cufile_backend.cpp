#include "backends.hpp"

#include <cuda_runtime.h>
#include <cufile.h>

#include <chrono>
#include <cerrno>
#include <cstdlib>
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

struct DriverGuard {
    bool open = false;
    DriverGuard() {
        cufile_check(cuFileDriverOpen(), "cuFileDriverOpen");
        open = true;
    }
    ~DriverGuard() { if (open) cuFileDriverClose(); }
    DriverGuard(const DriverGuard&) = delete;
    DriverGuard& operator=(const DriverGuard&) = delete;
};

struct FdGuard {
    int fd = -1;
    explicit FdGuard(int value) : fd(value) {}
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

struct HandleGuard {
    CUfileHandle_t handle{};
    bool registered = false;
    explicit HandleGuard(int fd) {
        CUfileDescr_t descriptor{};
        descriptor.handle.fd = fd;
        descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cufile_check(cuFileHandleRegister(&handle, &descriptor), "cuFileHandleRegister");
        registered = true;
    }
    ~HandleGuard() { if (registered) cuFileHandleDeregister(handle); }
    HandleGuard(const HandleGuard&) = delete;
    HandleGuard& operator=(const HandleGuard&) = delete;
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

bool env_true(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0;
}
}  // namespace

BenchmarkResult run_cufile(const BenchmarkOptions& options) {
    DriverGuard driver;

    const int raw_fd = ::open(options.path.c_str(), O_RDONLY | O_DIRECT);
    if (raw_fd < 0) {
        throw std::runtime_error(std::string("SKIPPED: open(O_DIRECT): ") + std::strerror(errno));
    }
    FdGuard fd(raw_fd);
    HandleGuard handle(fd.fd);
    DeviceBuffer device(options.bytes);

    const auto start = std::chrono::steady_clock::now();
    const ssize_t n = cuFileRead(handle.handle, device.ptr, options.bytes,
                                 static_cast<off_t>(options.offset), 0);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto end = std::chrono::steady_clock::now();

    if (n < 0) {
        throw std::runtime_error("SKIPPED: cuFileRead failed with code " + std::to_string(n));
    }
    if (static_cast<std::size_t>(n) != options.bytes) {
        throw std::runtime_error("cuFileRead returned fewer bytes than requested");
    }

    BenchmarkResult result;
    result.label = env_true("CUFILE_FORCE_COMPAT_MODE") ? "cufile-forced-compat"
                                                        : "cufile-auto-unverified-path";
    result.bytes = options.bytes;
    result.seconds = std::chrono::duration<double>(end - start).count();
    result.checksum = device_checksum(device.ptr, options.bytes);
    result.checksum_valid = true;
    return result;
}
