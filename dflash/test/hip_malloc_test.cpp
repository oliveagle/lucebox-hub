// Test hipMalloc with 15 GiB allocation
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

int main() {
    printf("[malloc_test] Starting hipMalloc test...\n");
    fflush(stdout);

    int device_count = 0;
    printf("[malloc_test] Calling hipGetDeviceCount()...\n");
    fflush(stdout);
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        fprintf(stderr, "[malloc_test] hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("[malloc_test] Found %d HIP devices\n", device_count);
    fflush(stdout);

    printf("[malloc_test] Calling hipSetDevice(0)...\n");
    fflush(stdout);
    err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "[malloc_test] hipSetDevice failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("[malloc_test] hipSetDevice(0) complete\n");
    fflush(stdout);

    // Try allocating 15 GiB (similar to test_dflash)
    const size_t alloc_size = 16091095040ULL;  // ~14.99 GiB
    printf("[malloc_test] Attempting hipMalloc of %zu bytes (%.2f GiB)...\n", alloc_size, alloc_size / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    void * dev_ptr = nullptr;
    err = hipMalloc(&dev_ptr, alloc_size);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (err != hipSuccess) {
        fprintf(stderr, "[malloc_test] hipMalloc FAILED: %s (took %lld ms)\n", hipGetErrorString(err), (long long)elapsed_ms);
        return 1;
    }

    printf("[malloc_test] hipMalloc SUCCESS! ptr=%p, took %lld ms (%.2f s)\n", dev_ptr, (long long)elapsed_ms, elapsed_ms / 1000.0);
    fflush(stdout);

    // Clean up
    printf("[malloc_test] Calling hipFree...\n");
    fflush(stdout);
    err = hipFree(dev_ptr);
    if (err != hipSuccess) {
        fprintf(stderr, "[malloc_test] hipFree failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("[malloc_test] hipFree complete\n");
    fflush(stdout);

    return 0;
}
