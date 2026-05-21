// Test mimicking load_target_gguf: allocate + cudaMemcpy from mmap
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
    printf("[load_test] Starting load_target_gguf simulation...\n");
    fflush(stdout);

    const char * model_path = "models/Qwen3.6-27B-Q4_K_M.gguf";
    printf("[load_test] Opening model: %s\n", model_path);
    fflush(stdout);

    // Open and mmap the model file
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    size_t file_len = st.st_size;
    printf("[load_test] File size: %zu bytes (%.2f GiB)\n", file_len, file_len / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);

    void * mm_addr = mmap(nullptr, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm_addr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    printf("[load_test] mmap successful at %p\n", mm_addr);
    fflush(stdout);
    close(fd);

    // Set device
    printf("[load_test] hipSetDevice(0)...\n");
    fflush(stdout);
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) { fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err)); return 1; }

    // Allocate 14.99 GiB
    const size_t alloc_total = 16091095040ULL;
    printf("[load_test] hipMalloc %.2f GiB...\n", alloc_total / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    void * dev_ptr = nullptr;
    err = hipMalloc(&dev_ptr, alloc_total);
    auto end = std::chrono::steady_clock::now();
    auto alloc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err)); munmap(mm_addr, file_len); return 1; }
    printf("[load_test] hipMalloc done in %lld ms, dev_ptr=%p\n", (long long)alloc_ms, dev_ptr);
    fflush(stdout);

    // Simulate ggml_backend_tensor_set for a few tensors
    // We'll copy a few chunks sequentially
    size_t total_copied = 0;
    const int n_test_copies = 50;
    size_t copy_size = alloc_total / 850;  // ~average tensor size

    printf("[load_test] Testing %d copies of ~%zu bytes from mmap to GPU...\n", n_test_copies, copy_size);
    fflush(stdout);

    for (int i = 0; i < n_test_copies; i++) {
        size_t src_off = (size_t)i * copy_size;
        size_t dst_off = (size_t)i * copy_size;

        auto cstart = std::chrono::steady_clock::now();
        err = hipMemcpy((char*)dev_ptr + dst_off, (const char*)mm_addr + src_off, copy_size, hipMemcpyDefault);
        auto cend = std::chrono::steady_clock::now();
        auto cms = std::chrono::duration_cast<std::chrono::milliseconds>(cend - cstart).count();

        if (err != hipSuccess) {
            fprintf(stderr, "[load_test] hipMemcpy %d FAILED: %s (took %lld ms)\n", i, hipGetErrorString(err), (long long)cms);
            hipFree(dev_ptr);
            munmap(mm_addr, file_len);
            return 1;
        }

        total_copied += copy_size;
        if (i % 10 == 9) {
            printf("[load_test] Copy %d: %.2f GiB copied, last copy took %lld ms\n", i + 1, total_copied / (1024.0 * 1024.0 * 1024.0), (long long)cms);
            fflush(stdout);
        }
    }

    printf("[load_test] All %d copies complete (%.2f GiB total)\n", n_test_copies, total_copied / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);

    // Clean up
    hipFree(dev_ptr);
    munmap(mm_addr, file_len);
    printf("[load_test] Cleanup complete\n");
    fflush(stdout);

    return 0;
}
