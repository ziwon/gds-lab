#include "backends.hpp"

#include <cuda_runtime.h>
#include <cufile.h>

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

namespace {
void cuda_check(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

std::string cufile_error_text(CUfileError_t status) {
    std::string text = CUFILE_ERRSTR(status.err);
    if (IS_CUDA_ERR(status)) {
        text += " (cuda driver error " + std::to_string(static_cast<int>(CU_FILE_CUDA_ERR(status))) + ")";
    }
    return text + " [code " + std::to_string(static_cast<int>(status.err)) + "]";
}

// cuFileRead returns a negative value on failure: a cuFile status when it is
// below the cuFile error base, otherwise a negated errno. Reporting the raw
// number alone makes every failure look the same.
std::string cufile_read_error_text(ssize_t n) {
    const int code = static_cast<int>(n);
    if (IS_CUFILE_ERR(code)) {
        return std::string(CUFILE_ERRSTR(code)) + " [code " + std::to_string(code) + "]";
    }
    return std::string(std::strerror(-code)) + " [errno " + std::to_string(-code) + "]";
}

void cufile_check(CUfileError_t status, const char* op) {
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error(std::string("SKIPPED: ") + op + ": " + cufile_error_text(status));
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

// cuFileBufRegister pins the device buffer for reuse across reads. Without it
// cuFile stages through its own preallocated bounce buffers, which is a
// different code path and a different number - so which one ran is recorded.
struct BufRegistration {
    const void* ptr = nullptr;
    bool active = false;
    std::string note;
    BufRegistration(const void* device, std::size_t bytes, bool requested) : ptr(device) {
        if (!requested) {
            note = "buf_register=off";
            return;
        }
        const CUfileError_t status = cuFileBufRegister(device, bytes, 0);
        if (status.err == CU_FILE_SUCCESS) {
            active = true;
            note = "buf_register=ok";
        } else {
            note = "buf_register=failed:" + cufile_error_text(status);
        }
    }
    ~BufRegistration() { if (active) cuFileBufDeregister(ptr); }
    BufRegistration(const BufRegistration&) = delete;
    BufRegistration& operator=(const BufRegistration&) = delete;
};

bool env_true(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0;
}
}  // namespace

BenchmarkResult run_cufile(const BenchmarkOptions& options) {
    if (options.read_chunk == 0) throw std::invalid_argument("--read-chunk must be > 0");

    DriverGuard driver;

    const int raw_fd = ::open(options.path.c_str(), O_RDONLY | O_DIRECT);
    if (raw_fd < 0) {
        throw std::runtime_error(std::string("SKIPPED: open(O_DIRECT): ") + std::strerror(errno));
    }
    FdGuard fd(raw_fd);
    HandleGuard handle(fd.fd);

    DeviceBuffer device(options.bytes);
    BufRegistration registration(device.ptr, options.bytes, options.buf_register);

    const std::size_t chunk = options.read_chunk;
    const std::size_t blocks = (options.bytes + chunk - 1) / chunk;
    std::vector<std::size_t> order(blocks);
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (options.random_access) {
        std::mt19937_64 rng(options.seed);
        std::shuffle(order.begin(), order.end(), rng);
    }

    const auto start = std::chrono::steady_clock::now();
    for (const std::size_t block : order) {
        const std::size_t block_offset = block * chunk;
        const std::size_t wanted = std::min(chunk, options.bytes - block_offset);
        std::size_t done = 0;
        while (done < wanted) {
            const ssize_t n = cuFileRead(handle.handle, device.ptr, wanted - done,
                                         static_cast<off_t>(options.offset + block_offset + done),
                                         static_cast<off_t>(block_offset + done));
            if (n < 0) {
                throw std::runtime_error("SKIPPED: cuFileRead: " + cufile_read_error_text(n));
            }
            if (n == 0) {
                throw std::runtime_error("unexpected EOF: file is smaller than --offset + --bytes");
            }
            done += static_cast<std::size_t>(n);
        }
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto end = std::chrono::steady_clock::now();

    BenchmarkResult result;
    result.label = std::string(env_true("CUFILE_FORCE_COMPAT_MODE") ? "cufile-forced-compat"
                                                                   : "cufile-auto-unverified-path") +
                   (options.random_access ? "-random" : "");
    result.bytes = options.bytes;
    result.seconds = std::chrono::duration<double>(end - start).count();
    // cuFile presents one call that covers storage read and device placement,
    // so there is no host-side read/copy boundary to time.
    result.split_valid = false;
    result.checksum = device_checksum(device.ptr, options.bytes);
    result.checksum_valid = true;
    result.note = registration.note;
    return result;
}
