#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                  << " -> " << hipGetErrorString(err) << std::endl;         \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {

    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    // Log only first few threads (otherwise output explodes)
    if (globalId < 5) {
        printf("[GPU] Block %d, Thread %d (global %d) running\n",
               (int)blockIdx.x, (int)threadIdx.x, globalId);
    }

    if (globalId < n) {
        c[globalId] = a[globalId] + b[globalId];

        if (globalId < 5) {
            printf("[GPU] a[%d]=%.1f + b[%d]=%.1f = %.1f\n",
                   globalId, a[globalId],
                   globalId, b[globalId],
                   c[globalId]);
        }
    }
}

int main() {

    std::cout << "==== CUDA Vector Add Debug Version ====\n";

    const int n = 16;   // small size for logging clarity
    const size_t bytes = n * sizeof(float);

    std::cout << "[HOST] Allocating host vectors...\n";

    std::vector<float> h_a(n), h_b(n), h_c(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 100 + i;
    }

    std::cout << "[HOST] Allocating device memory...\n";

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(hipMalloc(&d_a, bytes));
    CUDA_CHECK(hipMalloc(&d_b, bytes));
    CUDA_CHECK(hipMalloc(&d_c, bytes));

    std::cout << "[HOST] Copying data to device...\n";
    CUDA_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

    const int blockSize = 8;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    std::cout << "[HOST] Launching kernel with:\n";
    std::cout << "       Grid size  = " << gridSize << "\n";
    std::cout << "       Block size = " << blockSize << "\n";

    hipLaunchKernelGGL(vectorAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, n);

    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());

    std::cout << "[HOST] Copying results back to host...\n";
    CUDA_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    std::cout << "\n[HOST] Final Results:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "c[" << i << "] = " << h_c[i] << "\n";
    }

    CUDA_CHECK(hipFree(d_a));
    CUDA_CHECK(hipFree(d_b));
    CUDA_CHECK(hipFree(d_c));

    std::cout << "==== Done ====\n";

    return 0;
}