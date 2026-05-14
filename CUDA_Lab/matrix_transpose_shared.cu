// transpose_naive_vs_tiled.cu
//
// Runnable demo:
//  - Naive matrix transpose (global memory only)
//  - Tiled matrix transpose (shared memory + padding)
//  - Kernel timing with cudaEvent_t
//
// Compile (HIP for AMD iGPU):
//   hipcc -O3 matrix_transpose_shared.cu -o transpose_bench
// Compile (CUDA for NVIDIA GPU):
//   nvcc -O3 matrix_transpose_shared.cu -o transpose_bench
// Run:
//   ./transpose_bench 2048 2048

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    hipError_t err__ = (call);                                                \
    if (err__ != hipSuccess) {                                                \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__            \
                << " -> " << hipGetErrorString(err__) << std::endl;          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

#ifndef TRANSPOSE_TILE_DIM
#define TRANSPOSE_TILE_DIM 32
#endif

#ifndef TRANSPOSE_BLOCK_ROWS
#define TRANSPOSE_BLOCK_ROWS 8
#endif

// Input:  in[height][width]   row-major
// Output: out[width][height]  row-major (transpose)
__global__ void transposeNaive(float* __restrict__ out,
    const float* __restrict__ in,
    int width,
    int height) {
    const int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        const int yy = y + j;
        if (x < width && yy < height) {
            out[x * height + yy] = in[yy * width + x];
        }
    }
}

__global__ void transposeTiled(float* __restrict__ out,
    const float* __restrict__ in,
    int width,
    int height) {
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        const int yy = y + j;
        if (x < width && yy < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[yy * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        const int yy = y + j;
        if (x < height && yy < width) {
            out[yy * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

static void transposeCPU(const std::vector<float>& in,
    std::vector<float>& out,
    int width,
    int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            out[c * height + r] = in[r * width + c];
        }
    }
}

static void printDeviceInfo() {
    int dev = 0;
    CUDA_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    CUDA_CHECK(hipGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n\n";
}

int main(int argc, char** argv) {
    int height = 2048;
    int width = 2048;
    if (argc == 3) {
        height = std::stoi(argv[1]);
        width = std::stoi(argv[2]);
    }
    else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [HEIGHT WIDTH]\n";
        return 1;
    }

    printDeviceInfo();
    std::cout << "Transpose input: " << height << "x" << width
        << " -> output: " << width << "x" << height << "\n";
    std::cout << "TILE_DIM: " << TRANSPOSE_TILE_DIM
        << ", BLOCK_ROWS: " << TRANSPOSE_BLOCK_ROWS << "\n\n";

    const size_t numIn = (size_t)height * (size_t)width;
    const size_t numOut = (size_t)width * (size_t)height;
    const size_t bytesIn = numIn * sizeof(float);
    const size_t bytesOut = numOut * sizeof(float);

    std::vector<float> hIn(numIn);
    std::vector<float> hOutNaive(numOut, 0.0f);
    std::vector<float> hOutTiled(numOut, 0.0f);
    std::vector<float> hOutCPU(numOut, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hIn) x = dist(rng);

    float* dIn = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(hipMalloc(&dIn, bytesIn));
    CUDA_CHECK(hipMalloc(&dOut, bytesOut));
    CUDA_CHECK(hipMemcpy(dIn, hIn.data(), bytesIn, hipMemcpyHostToDevice));

    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    dim3 grid(divUp(width, TRANSPOSE_TILE_DIM), divUp(height, TRANSPOSE_TILE_DIM));

    std::cout << "Launch: grid(" << grid.x << "," << grid.y
        << ") block(" << block.x << "," << block.y << ")\n\n";

    const int warmup = 3;
    const int iters = 50;

    auto timeKernel = [&](auto&& launch) -> float {
        for (int i = 0; i < warmup; i++) launch();
        CUDA_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        CUDA_CHECK(hipEventCreate(&start));
        CUDA_CHECK(hipEventCreate(&stop));

        CUDA_CHECK(hipEventRecord(start));
        for (int i = 0; i < iters; i++) launch();
        CUDA_CHECK(hipEventRecord(stop));
        CUDA_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(hipEventDestroy(start));
        CUDA_CHECK(hipEventDestroy(stop));
        return ms / iters;
        };

    auto launchNaive = [&]() {
        hipLaunchKernelGGL(transposeNaive, grid, block, 0, 0, dOut, dIn, width, height);
        };

    auto launchTiled = [&]() {
        hipLaunchKernelGGL(transposeTiled, grid, block, 0, 0, dOut, dIn, width, height);
        };

    CUDA_CHECK(hipMemset(dOut, 0, bytesOut));
    float naiveMs = timeKernel(launchNaive);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpy(hOutNaive.data(), dOut, bytesOut, hipMemcpyDeviceToHost));

    CUDA_CHECK(hipMemset(dOut, 0, bytesOut));
    float tiledMs = timeKernel(launchTiled);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpy(hOutTiled.data(), dOut, bytesOut, hipMemcpyDeviceToHost));


    std::cout << std::fixed << std::setprecision(9);
    std::cout << "=== Transpose Timing (Kernel-only avg) ===\n";
    std::cout << "  Naive : " << naiveMs << " ms\n";
    std::cout << "  Tiled : " << tiledMs << " ms\n";


    CUDA_CHECK(hipFree(dIn));
    CUDA_CHECK(hipFree(dOut));
    return 0;
}
