// Cross-GPU device memory copy via pinned host staging (no CUDA peer access).

#include "internal.h"

#include <cuda_runtime.h>

#include <mutex>

namespace {

std::mutex g_pin_mu;
void *     g_pin     = nullptr;
size_t     g_pin_cap = 0;

} // namespace

bool dflash_cuda_copy_between_devices(int src_dev, const void * src,
                                      int dst_dev, void * dst, size_t nbytes,
                                      cudaStream_t stream) {
    if (nbytes == 0) {
        return true;
    }
    if (src_dev == dst_dev) {
        cudaError_t err = cudaSetDevice(src_dev);
        if (err != cudaSuccess) {
            return false;
        }
        err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            return false;
        }
        if (stream) {
            return cudaStreamSynchronize(stream) == cudaSuccess;
        }
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    void * pin = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_pin_mu);
        if (g_pin_cap < nbytes) {
            if (g_pin) {
                cudaFreeHost(g_pin);
                g_pin     = nullptr;
                g_pin_cap = 0;
            }
            cudaError_t err = cudaMallocHost(&g_pin, nbytes);
            if (err != cudaSuccess) {
                return false;
            }
            g_pin_cap = nbytes;
        }
        pin = g_pin;
    }

    cudaError_t err = cudaSetDevice(src_dev);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMemcpyAsync(pin, src, nbytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaSetDevice(dst_dev);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMemcpyAsync(dst, pin, nbytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return false;
    }
    if (stream) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            return false;
        }
    } else {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }
    }
    return true;
}
