// Minimal HIP test to reproduce the hang issue
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("[minimal] Starting HIP test...\n");
    fflush(stdout);

    int device_count = 0;
    printf("[minimal] Calling hipGetDeviceCount()...\n");
    fflush(stdout);
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        fprintf(stderr, "[minimal] hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("[minimal] Found %d HIP devices\n", device_count);
    fflush(stdout);

    for (int id = 0; id < device_count; ++id) {
        printf("[minimal] Processing device %d...\n", id);
        fflush(stdout);

        hipDeviceProp_t prop;
        printf("[minimal] Calling hipGetDeviceProperties() for device %d...\n", id);
        fflush(stdout);
        err = hipGetDeviceProperties(&prop, id);
        if (err != hipSuccess) {
            fprintf(stderr, "[minimal] hipGetDeviceProperties failed for device %d: %s\n", id, hipGetErrorString(err));
            continue;
        }
        printf("[minimal] Device %d: %s (gcnArchName: %s)\n", id, prop.name, prop.gcnArchName);
        printf("[minimal]   VRAM: %zu MiB, SM: %d, Wave Size: %d\n",
               (size_t)(prop.totalGlobalMem / (1024 * 1024)),
               prop.multiProcessorCount, prop.warpSize);
        fflush(stdout);

        printf("[minimal] Calling hipSetDevice(%d)...\n", id);
        fflush(stdout);
        err = hipSetDevice(id);
        if (err != hipSuccess) {
            fprintf(stderr, "[minimal] hipSetDevice failed: %s\n", hipGetErrorString(err));
            continue;
        }
        printf("[minimal] hipSetDevice(%d) complete\n", id);
        fflush(stdout);

        // Skip peer access for single device
        if (device_count > 1) {
            for (int id_other = 0; id_other < device_count; ++id_other) {
                if (id == id_other) continue;
                printf("[minimal] Checking peer access %d -> %d...\n", id, id_other);
                fflush(stdout);
                int can_access = 0;
                err = hipDeviceCanAccessPeer(&can_access, id, id_other);
                if (err != hipSuccess) {
                    fprintf(stderr, "[minimal] hipDeviceCanAccessPeer failed: %s\n", hipGetErrorString(err));
                    continue;
                }
                printf("[minimal] Can access peer: %d\n", can_access);
                fflush(stdout);
                if (can_access) {
                    printf("[minimal] Calling hipDeviceEnablePeerAccess()...\n");
                    fflush(stdout);
                    err = hipDeviceEnablePeerAccess(id_other, 0);
                    if (err != hipSuccess) {
                        fprintf(stderr, "[minimal] hipDeviceEnablePeerAccess failed: %s\n", hipGetErrorString(err));
                        continue;
                    }
                    printf("[minimal] Peer access enabled\n");
                    fflush(stdout);
                }
            }
        }
    }

    printf("[minimal] All steps completed successfully!\n");
    fflush(stdout);
    return 0;
}
