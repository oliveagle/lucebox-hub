// Test large synchronous hipMemcpy from mmap
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

int main() {
    printf("[large_test] Starting large hipMemcpy test...\n");
    fflush(stdout);

    // Set device
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) { fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err)); return 1; }

    // Allocate ~1 GiB on device
    const size_t alloc_size = 1024ULL * 1024 * 1024;  // 1 GiB
    printf("[large_test] hipMalloc %zu bytes...\n", alloc_size);
    fflush(stdout);

    void * dev_ptr = nullptr;
    err = hipMalloc(&dev_ptr, alloc_size);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err)); return 1; }
    printf("[large_test] hipMalloc done\n");
    fflush(stdout);

    // Allocate host memory
    printf("[large_test] Allocating host memory...\n");
    fflush(stdout);
    void * host_ptr = malloc(alloc_size);
    if (!host_ptr) { perror("malloc"); hipFree(dev_ptr); return 1; }
    memset(host_ptr, 0x42, alloc_size);
    printf("[large_test] Host memory ready\n");
    fflush(stdout);

    // Test synchronous hipMemcpy
    printf("[large_test] Calling hipMemcpy (synchronous)...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    err = hipMemcpy(dev_ptr, host_ptr, alloc_size, hipMemcpyHostToDevice);
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (err != hipSuccess) {
        fprintf(stderr, "[large_test] hipMemcpy FAILED: %s (took %lld ms)\n", hipGetErrorString(err), (long long)ms);
        hipFree(dev_ptr);
        free(host_ptr);
        return 1;
    }

    printf("[large_test] hipMemcpy SUCCESS: %.2f GiB in %lld ms (%.2f GB/s)\n",
           alloc_size / (1024.0 * 1024.0 * 1024.0),
           (long long)ms,
           alloc_size / (1024.0 * 1024.0 * 1024.0) / (ms / 1000.0));
    fflush(stdout);

    // Test with mmap'd memory
    const char * model_path = "models/Qwen3.6-27B-Q4_K_M.gguf";
    printf("[large_test] Opening model: %s\n", model_path);
    fflush(stdout);

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { perror("open"); hipFree(dev_ptr); free(host_ptr); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); hipFree(dev_ptr); free(host_ptr); return 1; }

    void * mm_addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm_addr == MAP_FAILED) { perror("mmap"); close(fd); hipFree(dev_ptr); free(host_ptr); return 1; }
    close(fd);

    printf("[large_test] mmap opened, attempting hipMemcpy from mmap...\n");
    fflush(stdout);

    size_t copy_size = (alloc_size < (size_t)st.st_size) ? alloc_size : (size_t)st.st_size;
    start = std::chrono::steady_clock::now();
    err = hipMemcpy(dev_ptr, mm_addr, copy_size, hipMemcpyHostToDevice);
    end = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (err != hipSuccess) {
        fprintf(stderr, "[large_test] hipMemcpy from mmap FAILED: %s (took %lld ms)\n", hipGetErrorString(err), (long long)ms);
    } else {
        printf("[large_test] hipMemcpy from mmap SUCCESS: %.2f GiB in %lld ms\n",
               copy_size / (1024.0 * 1024.0 * 1024.0), (long long)ms);
        fflush(stdout);
    }

    munmap(mm_addr, st.st_size);
    hipFree(dev_ptr);
    free(host_ptr);
    printf("[large_test] Cleanup complete\n");
    fflush(stdout);

    return 0;
}
