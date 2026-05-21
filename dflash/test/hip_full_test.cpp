// Test hipMemcpy with malloc'd vs mmap'd memory
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
    printf("[test] Starting...\n"); fflush(stdout);
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) { fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err)); return 1; }

    // Allocate small device buffer
    const size_t alloc_size = 10ULL * 1024 * 1024;  // 10 MiB
    printf("[test] hipMalloc 10 MiB...\n"); fflush(stdout);
    void * dev_ptr = nullptr;
    err = hipMalloc(&dev_ptr, alloc_size);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err)); return 1; }

    // Test 1: hipMemcpy with malloc'd memory
    void * malloc_ptr = malloc(alloc_size);
    memset(malloc_ptr, 0x42, alloc_size);

    printf("[test] Test 1: hipMemcpy with malloc memory (10 MiB)...\n"); fflush(stdout);
    auto start = std::chrono::steady_clock::now();
    err = hipMemcpy(dev_ptr, malloc_ptr, alloc_size, hipMemcpyHostToDevice);
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[test] hipMemcpy(malloc) %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

    if (err == hipSuccess) {
        // Test 2: hipMemcpy with mmap'd memory
        const char * model_path = "models/Qwen3.6-27B-Q4_K_M.gguf";
        printf("[test] Opening model...\n"); fflush(stdout);
        int fd = open(model_path, O_RDONLY);
        if (fd < 0) { perror("open"); }
        else {
            struct stat st;
            fstat(fd, &st);
            printf("[test] File size: %.2f GiB\n", st.st_size / (1024.0 * 1024.0 * 1024.0)); fflush(stdout);

            void * mm_addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mm_addr == MAP_FAILED) { perror("mmap"); }
            else {
                printf("[test] Test 2: hipMemcpy with mmap memory (10 MiB)...\n"); fflush(stdout);
                start = std::chrono::steady_clock::now();
                err = hipMemcpy(dev_ptr, mm_addr, alloc_size, hipMemcpyHostToDevice);
                end = std::chrono::steady_clock::now();
                ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                printf("[test] hipMemcpy(mmap) %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

                // Test 3: hipMemcpy with mmap memory - small chunk at file offset
                const size_t offset = 1024;  // skip GGUF header
                printf("[test] Test 3: hipMemcpy with mmap at offset %zu (10 MiB)...\n", offset); fflush(stdout);
                start = std::chrono::steady_clock::now();
                err = hipMemcpy(dev_ptr, (char*)mm_addr + offset, alloc_size, hipMemcpyHostToDevice);
                end = std::chrono::steady_clock::now();
                ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                printf("[test] hipMemcpy(mmap+offset) %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

                // Test 4: hipMemcpyHtoD with mmap'd memory
                printf("[test] Test 4: hipMemcpyHtoD with mmap memory (10 MiB)...\n"); fflush(stdout);
                start = std::chrono::steady_clock::now();
                err = hipMemcpyHtoD((hipDeviceptr_t)dev_ptr, (char*)mm_addr + offset, alloc_size);
                end = std::chrono::steady_clock::now();
                ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                printf("[test] hipMemcpyHtoD(mmap) %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);

                munmap(mm_addr, st.st_size);
            }
            close(fd);
        }
    }

    // Test 5: hipHostMalloc + hipMemcpy (pinned memory)
    void * pinned_ptr = nullptr;
    printf("[test] Test 5: hipHostMalloc + hipMemcpy (10 MiB)...\n"); fflush(stdout);
    err = hipHostMalloc(&pinned_ptr, alloc_size, hipHostMallocDefault);
    if (err == hipSuccess) {
        memset(pinned_ptr, 0x42, alloc_size);
        start = std::chrono::steady_clock::now();
        err = hipMemcpy(dev_ptr, pinned_ptr, alloc_size, hipMemcpyHostToDevice);
        end = std::chrono::steady_clock::now();
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("[test] hipMemcpy(pinned) %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);
        hipHostFree(pinned_ptr);
    } else {
        printf("[test] hipHostMalloc failed: %s\n", hipGetErrorString(err)); fflush(stdout);
    }

    // Test 6: hipMallocManaged
    void * managed_ptr = nullptr;
    printf("[test] Test 6: hipMallocManaged + hipMemcpy (10 MiB)...\n"); fflush(stdout);
    err = hipMallocManaged(&managed_ptr, alloc_size);
    if (err == hipSuccess) {
        memset(managed_ptr, 0x43, alloc_size);
        start = std::chrono::steady_clock::now();
        err = hipMemcpy(dev_ptr, managed_ptr, alloc_size, hipMemcpyHostToDevice);
        end = std::chrono::steady_clock::now();
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("[test] hipMemcpy(managed) %s (took %lld ms)\n", err == hipSuccess ? "SUCCESS" : "FAILED", (long long)ms); fflush(stdout);
        hipFree(managed_ptr);
    } else {
        printf("[test] hipMallocManaged failed: %s\n", hipGetErrorString(err)); fflush(stdout);
    }

    hipFree(dev_ptr);
    free(malloc_ptr);
    printf("[test] Done\n");
    return 0;
}
