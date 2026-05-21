#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

int main() {
    hipSetDevice(0);

    const size_t alloc_size = 100ULL * 1024 * 1024;  // 100 MiB
    printf("hipMalloc %zu bytes...\n", alloc_size); fflush(stdout);
    void * dev_ptr = nullptr;
    hipError_t err = hipMalloc(&dev_ptr, alloc_size);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err)); return 1; }

    void * host_ptr = malloc(alloc_size);
    memset(host_ptr, 0x42, alloc_size);

    // Test 1: hipMemcpyAsync with NULL stream (0)
    printf("Test 1: hipMemcpyAsync(NULL stream)...\n"); fflush(stdout);
    auto start = std::chrono::steady_clock::now();
    err = hipMemcpyAsync(dev_ptr, host_ptr, alloc_size, hipMemcpyHostToDevice, 0);
    printf("hipMemcpyAsync returned: %s\n", err == hipSuccess ? "SUCCESS" : "FAILED"); fflush(stdout);

    printf("hipDeviceSynchronize()...\n"); fflush(stdout);
    err = hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("hipDeviceSynchronize() %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

    // Test 2: hipMemcpyAsync with explicit stream
    hipStream_t stream = nullptr;
    printf("hipStreamCreate()...\n"); fflush(stdout);
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) { fprintf(stderr, "hipStreamCreate failed: %s\n", hipGetErrorString(err)); hipFree(dev_ptr); free(host_ptr); return 1; }

    start = std::chrono::steady_clock::now();
    printf("hipMemcpyAsync(explicit stream)...\n"); fflush(stdout);
    err = hipMemcpyAsync(dev_ptr, host_ptr, alloc_size, hipMemcpyHostToDevice, stream);
    printf("hipMemcpyAsync returned: %s\n", err == hipSuccess ? "SUCCESS" : "FAILED"); fflush(stdout);

    printf("hipStreamSynchronize()...\n"); fflush(stdout);
    err = hipStreamSynchronize(stream);
    end = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("hipStreamSynchronize() %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

    hipStreamDestroy(stream);

    // Test 3: hipMemcpyHtoD (driver API)
    printf("hipMemcpyHtoD(100 bytes)...\n"); fflush(stdout);
    start = std::chrono::steady_clock::now();
    err = hipMemcpyHtoD(dev_ptr, host_ptr, 100);
    end = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("hipMemcpyHtoD %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

    // Cleanup
    hipFree(dev_ptr);
    free(host_ptr);
    printf("Done\n");
    return 0;
}
