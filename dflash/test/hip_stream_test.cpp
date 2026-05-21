// Test if hipStreamPerThread is working on this device
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

int main() {
    printf("[stream_test] Starting hipStreamPerThread test...\n");
    fflush(stdout);

    // Set device
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) { fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err)); return 1; }
    printf("[stream_test] Device 0 set\n");
    fflush(stdout);

    // Allocate memory
    const size_t alloc_size = 1024 * 1024;  // 1 MiB
    void * dev_ptr = nullptr;
    err = hipMalloc(&dev_ptr, alloc_size);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err)); return 1; }
    printf("[stream_test] hipMalloc %zu bytes done\n", alloc_size);
    fflush(stdout);

    // Allocate host memory
    void * host_ptr = malloc(alloc_size);
    memset(host_ptr, 0x42, alloc_size);

    // Test 1: hipMemcpy (synchronous)
    printf("[stream_test] Test 1: hipMemcpy (synchronous)...\n");
    fflush(stdout);
    auto start = std::chrono::steady_clock::now();
    err = hipMemcpy(dev_ptr, host_ptr, alloc_size, hipMemcpyHostToDevice);
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[stream_test] hipMemcpy %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms);
    fflush(stdout);

    // Test 2: hipStreamPerThread with hipMemcpyAsync
    printf("[stream_test] Test 2: hipMemcpyAsync + hipStreamSynchronize(hipStreamPerThread)...\n");
    fflush(stdout);
    start = std::chrono::steady_clock::now();
    err = hipMemcpyAsync(dev_ptr, host_ptr, alloc_size, hipMemcpyHostToDevice, hipStreamPerThread);
    printf("[stream_test] hipMemcpyAsync returned: %s\n", err == hipSuccess ? "SUCCESS" : "FAILED");
    fflush(stdout);
    if (err != hipSuccess) {
        fprintf(stderr, "[stream_test] hipMemcpyAsync failed: %s\n", hipGetErrorString(err));
        hipFree(dev_ptr);
        free(host_ptr);
        return 1;
    }
    printf("[stream_test] About to call hipStreamSynchronize(hipStreamPerThread)...\n");
    fflush(stdout);
    err = hipStreamSynchronize(hipStreamPerThread);
    end = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[stream_test] hipStreamSynchronize(hipStreamPerThread) %s (took %lld ms)\n",
           err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms);
    fflush(stdout);

    // Test 3: Explicit stream
    printf("[stream_test] Test 3: hipStreamCreate + hipMemcpyAsync + hipStreamSynchronize...\n");
    fflush(stdout);
    hipStream_t explicit_stream;
    err = hipStreamCreate(&explicit_stream);
    if (err != hipSuccess) { fprintf(stderr, "hipStreamCreate failed: %s\n", hipGetErrorString(err)); hipFree(dev_ptr); free(host_ptr); return 1; }
    start = std::chrono::steady_clock::now();
    err = hipMemcpyAsync(dev_ptr, host_ptr, alloc_size, hipMemcpyHostToDevice, explicit_stream);
    printf("[stream_test] hipMemcpyAsync returned: %s\n", err == hipSuccess ? "SUCCESS" : "FAILED");
    fflush(stdout);
    if (err != hipSuccess) {
        hipStreamDestroy(explicit_stream);
        hipFree(dev_ptr);
        free(host_ptr);
        return 1;
    }
    err = hipStreamSynchronize(explicit_stream);
    end = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[stream_test] hipStreamSynchronize(explicit) %s (took %lld ms)\n",
           err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms);
    fflush(stdout);
    hipStreamDestroy(explicit_stream);

    // Cleanup
    hipFree(dev_ptr);
    free(host_ptr);
    printf("[stream_test] Cleanup complete\n");
    fflush(stdout);

    return 0;
}
