// matrix_transpose.cu
//
// Lab 1 – Assignment 2
// Transposes a matrix A (rows×cols) on the GPU,
// producing A^T (cols×rows).
//
// Compile:
//   nvcc -O3 matrix_transpose.cu -o matrix_transpose
// Run:
//   ./matrix_transpose           (default 1024×1024)
//   ./matrix_transpose 2048 2048

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <random>
#include <iomanip>

// --------------------------------------------------------------------------
// Error-checking macro
// --------------------------------------------------------------------------
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

#define TILE_DIM   32    // tile width == tile height
#define BLOCK_ROWS  8    // number of rows handled per block pass

// --------------------------------------------------------------------------
// Naive transpose kernel  (global memory only)
//
// Each thread reads one element from in[row][col] and writes it to
// out[col][row].  The read is coalesced (consecutive threads read
// consecutive columns) but the write is not (strided by `height`).
//
// in  : row-major height×width
// out : row-major width×height  (transposed)
// --------------------------------------------------------------------------
__global__ void transposeNaive(float* __restrict__ out,
                                const float* __restrict__ in,
                                int width, int height)
{
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;  // column in input
    const int y = blockIdx.y * TILE_DIM + threadIdx.y;  // row    in input

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        const int yy = y + j;
        if (x < width && yy < height)
            out[x * height + yy] = in[yy * width + x];
    }
}

// --------------------------------------------------------------------------
// CPU reference
// --------------------------------------------------------------------------
static void transposeCPU(const std::vector<float>& in,
                         std::vector<float>&       out,
                         int width, int height)
{
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c)
            out[c * height + r] = in[r * width + c];
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int height = 1024, width = 1024;
    if (argc == 3) {
        height = std::stoi(argv[1]);
        width  = std::stoi(argv[2]);
    }

    std::cout << "=== CUDA Matrix Transpose  ("
              << height << "x" << width << ") -> ("
              << width  << "x" << height << ") ===\n\n";

    // ------------------------------------------------------------------
    // Host data
    // ------------------------------------------------------------------
    const size_t numIn  = (size_t)height * width;
    const size_t numOut = (size_t)width  * height;

    std::vector<float> hIn(numIn), hOut(numOut, 0.0f), hRef(numOut, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hIn) v = dist(rng);

    // ------------------------------------------------------------------
    // Device memory
    // ------------------------------------------------------------------
    float *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(hipMalloc(&dIn,  numIn  * sizeof(float)));
    CUDA_CHECK(hipMalloc(&dOut, numOut * sizeof(float)));
    CUDA_CHECK(hipMemcpy(dIn, hIn.data(), numIn * sizeof(float), hipMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Launch configuration
    // ------------------------------------------------------------------
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(divUp(width, TILE_DIM), divUp(height, TILE_DIM));

    std::cout << "Block: (" << block.x << ", " << block.y << ")\n";
    std::cout << "Grid : (" << grid.x  << ", " << grid.y  << ")\n\n";

    // ------------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------------
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));

    // Warm-up
    hipLaunchKernelGGL(transposeNaive, grid, block, 0, 0, dOut, dIn, width, height);
    CUDA_CHECK(hipDeviceSynchronize());

    CUDA_CHECK(hipMemset(dOut, 0, numOut * sizeof(float)));
    CUDA_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(transposeNaive, grid, block, 0, 0, dOut, dIn, width, height);
    CUDA_CHECK(hipEventRecord(stop));
    CUDA_CHECK(hipEventSynchronize(stop));
    CUDA_CHECK(hipGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU transpose time: " << ms << " ms\n\n";

    // ------------------------------------------------------------------
    // Copy result back and verify
    // ------------------------------------------------------------------
    CUDA_CHECK(hipMemcpy(hOut.data(), dOut, numOut * sizeof(float), hipMemcpyDeviceToHost));

    transposeCPU(hIn, hRef, width, height);

    float maxErr = 0.0f;
    for (size_t i = 0; i < numOut; ++i)
        maxErr = std::max(maxErr, std::fabs(hOut[i] - hRef[i]));

    std::cout << "Max absolute error vs CPU: " << maxErr << "\n";
    if (maxErr < 1e-5f)
        std::cout << "PASS – transpose is correct.\n\n";
    else
        std::cout << "FAIL – transpose differs from CPU reference!\n\n";

    // ------------------------------------------------------------------
    // Print a small corner
    // ------------------------------------------------------------------
    int pr = std::min(height, 4), pc = std::min(width, 4);
    std::cout << "Top-left " << pc << "x" << pr << " of A^T:\n";
    for (int r = 0; r < pc; ++r) {
        for (int c = 0; c < pr; ++c)
            std::cout << std::setw(10) << hOut[r * height + c] << " ";
        std::cout << "\n";
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    CUDA_CHECK(hipFree(dIn));
    CUDA_CHECK(hipFree(dOut));
    CUDA_CHECK(hipEventDestroy(start));
    CUDA_CHECK(hipEventDestroy(stop));

    std::cout << "\n=== Done ===\n";
    return 0;
}
