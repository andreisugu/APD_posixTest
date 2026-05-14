// conv2d_shared.cu
//
// Lab 2 – Assignment 1
// 2D convolution on the GPU with "valid" boundary condition.
//
//   output_rows = input_rows - kernel_rows + 1
//   output_cols = input_cols - kernel_cols + 1
//
//   output[i][j] = sum_{m,n} input[i+m][j+n] * kernel[m][n]
//
// Two implementations are provided:
//   1. Naive  – each thread reads directly from global memory.
//   2. Tiled  – threads cooperatively load an input halo tile into shared
//               memory before computing, reducing global memory traffic.
//
// Compile:
//   nvcc -O3 conv2d_shared.cu -o conv2d_shared
// Run:
//   ./conv2d_shared            (default 1024×1024 input, 5×5 kernel)
//   ./conv2d_shared 2048 2048 7 7

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
    hipError_t err__ = (call);                                               \
    if (err__ != hipSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__           \
                << " -> " << hipGetErrorString(err__) << std::endl;         \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------
#define BLOCK_W 16    // output tile width  (threads per block in x)
#define BLOCK_H 16    // output tile height (threads per block in y)

// Maximum kernel radius supported by the constant-memory path
// (kernel stored in constant memory for fast broadcast reads).
// We support kernels up to 15×15 (radius 7).
#define MAX_KERNEL_RADIUS 7
#define MAX_KERNEL_SIZE   ((2*MAX_KERNEL_RADIUS+1)*(2*MAX_KERNEL_RADIUS+1))

// Kernel (filter) weights in constant memory – fastest for read-only broadcast
__constant__ float c_kernel[MAX_KERNEL_SIZE];

// --------------------------------------------------------------------------
// Naive 2-D convolution kernel  (global memory only)
//
// Each output thread independently reads all input[i+m][j+n] from global
// memory and accumulates the dot product with the filter.
// --------------------------------------------------------------------------
__global__ void conv2dNaive(const float* __restrict__ input,
                             float*       __restrict__ output,
                             int inRows,  int inCols,
                             int kRows,   int kCols,
                             int outRows, int outCols)
{
    int outCol = blockIdx.x * BLOCK_W + threadIdx.x;
    int outRow = blockIdx.y * BLOCK_H + threadIdx.y;

    if (outRow >= outRows || outCol >= outCols) return;

    float sum = 0.0f;
    for (int m = 0; m < kRows; ++m)
        for (int n = 0; n < kCols; ++n)
            sum += input[(outRow + m) * inCols + (outCol + n)]
                 * c_kernel[m * kCols + n];

    output[outRow * outCols + outCol] = sum;
}

// --------------------------------------------------------------------------
// Tiled 2-D convolution kernel  (shared memory)
//
// Strategy:
//   Each block of BLOCK_H × BLOCK_W output threads loads a larger halo tile
//   of the input into shared memory.  The halo extends (kRows-1) rows below
//   and (kCols-1) columns to the right of the output tile so every thread
//   has all the input values it needs without hitting global memory again.
//
// Shared memory tile size:
//   (BLOCK_H + kRows - 1) × (BLOCK_W + kCols - 1)
//
// To keep shared memory allocation static we cap the halo at MAX_KERNEL_RADIUS
// on each side, giving a maximum tile of:
//   (BLOCK_H + 2*MAX_KERNEL_RADIUS) × (BLOCK_W + 2*MAX_KERNEL_RADIUS)
// --------------------------------------------------------------------------
#define SMEM_H (BLOCK_H + 2*MAX_KERNEL_RADIUS)
#define SMEM_W (BLOCK_W + 2*MAX_KERNEL_RADIUS)

__global__ void conv2dTiled(const float* __restrict__ input,
                             float*       __restrict__ output,
                             int inRows,  int inCols,
                             int kRows,   int kCols,
                             int outRows, int outCols)
{
    // Shared memory tile – large enough for the maximum supported kernel size
    __shared__ float smem[SMEM_H][SMEM_W];

    // Top-left corner of the input region this block is responsible for
    int inStartRow = blockIdx.y * BLOCK_H;
    int inStartCol = blockIdx.x * BLOCK_W;

    // Total elements to load into shared memory
    const int smemRows = kRows - 1 + BLOCK_H;   // actual halo height
    const int smemCols = kCols - 1 + BLOCK_W;   // actual halo width
    const int numElems = smemRows * smemCols;
    const int numThreads = BLOCK_H * BLOCK_W;
    int tid = threadIdx.y * BLOCK_W + threadIdx.x;

    // Cooperatively fill the shared-memory tile.
    // Each thread may load more than one element if the tile is larger than
    // the block.
    for (int idx = tid; idx < numElems; idx += numThreads) {
        int smRow = idx / smemCols;
        int smCol = idx % smemCols;
        int glRow = inStartRow + smRow;
        int glCol = inStartCol + smCol;

        if (glRow < inRows && glCol < inCols)
            smem[smRow][smCol] = input[glRow * inCols + glCol];
        else
            smem[smRow][smCol] = 0.0f;   // zero-padding for out-of-bounds
    }

    // All threads must finish loading before any thread starts computing
    __syncthreads();

    // Compute one output element per thread
    int outRow = blockIdx.y * BLOCK_H + threadIdx.y;
    int outCol = blockIdx.x * BLOCK_W + threadIdx.x;

    if (outRow >= outRows || outCol >= outCols) return;

    float sum = 0.0f;
    for (int m = 0; m < kRows; ++m)
        for (int n = 0; n < kCols; ++n)
            sum += smem[threadIdx.y + m][threadIdx.x + n]
                 * c_kernel[m * kCols + n];

    output[outRow * outCols + outCol] = sum;
}

// --------------------------------------------------------------------------
// CPU reference  (for correctness verification)
// --------------------------------------------------------------------------
static void conv2dCPU(const std::vector<float>& input,
                      const std::vector<float>& kernel,
                      std::vector<float>&       output,
                      int inRows,  int inCols,
                      int kRows,   int kCols,
                      int outRows, int outCols)
{
    for (int i = 0; i < outRows; ++i)
        for (int j = 0; j < outCols; ++j) {
            float s = 0.0f;
            for (int m = 0; m < kRows; ++m)
                for (int n = 0; n < kCols; ++n)
                    s += input[(i + m) * inCols + (j + n)] * kernel[m * kCols + n];
            output[i * outCols + j] = s;
        }
}

// --------------------------------------------------------------------------
// Helper: time a kernel over `iters` iterations (with `warmup` warm-up runs)
// --------------------------------------------------------------------------
template<typename Fn>
static float timeKernel(Fn launch, int warmup = 3, int iters = 20)
{
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));

    CUDA_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(hipEventRecord(stop));
    CUDA_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(hipEventDestroy(start));
    CUDA_CHECK(hipEventDestroy(stop));
    return ms / iters;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // --- dimensions -------------------------------------------------------
    int inRows = 1024, inCols = 1024, kRows = 5, kCols = 5;
    if (argc == 5) {
        inRows = std::stoi(argv[1]);
        inCols = std::stoi(argv[2]);
        kRows  = std::stoi(argv[3]);
        kCols  = std::stoi(argv[4]);
    }

    if (kRows > 2 * MAX_KERNEL_RADIUS + 1 || kCols > 2 * MAX_KERNEL_RADIUS + 1) {
        std::cerr << "Kernel too large (max "
                  << 2*MAX_KERNEL_RADIUS+1 << "x" << 2*MAX_KERNEL_RADIUS+1 << ")\n";
        return 1;
    }

    const int outRows = inRows - kRows + 1;
    const int outCols = inCols - kCols + 1;

    std::cout << "=== CUDA 2D Convolution (valid) ===\n";
    std::cout << "Input : " << inRows << "x" << inCols << "\n";
    std::cout << "Kernel: " << kRows  << "x" << kCols  << "\n";
    std::cout << "Output: " << outRows << "x" << outCols << "\n\n";

    // --- host data --------------------------------------------------------
    const size_t szIn  = (size_t)inRows  * inCols;
    const size_t szK   = (size_t)kRows   * kCols;
    const size_t szOut = (size_t)outRows * outCols;

    std::vector<float> hIn(szIn), hKernel(szK);
    std::vector<float> hOutNaive(szOut, 0.0f);
    std::vector<float> hOutTiled(szOut, 0.0f);
    std::vector<float> hRef(szOut, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hIn)     v = dist(rng);
    for (auto& v : hKernel) v = dist(rng);

    // --- upload kernel to constant memory ---------------------------------
    CUDA_CHECK(hipMemcpyToSymbol(c_kernel, hKernel.data(), szK * sizeof(float)));

    // --- device memory ----------------------------------------------------
    float *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(hipMalloc(&dIn,  szIn  * sizeof(float)));
    CUDA_CHECK(hipMalloc(&dOut, szOut * sizeof(float)));
    CUDA_CHECK(hipMemcpy(dIn, hIn.data(), szIn * sizeof(float), hipMemcpyHostToDevice));

    // --- launch config ----------------------------------------------------
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid(divUp(outCols, BLOCK_W), divUp(outRows, BLOCK_H));

    std::cout << "Block: (" << block.x << ", " << block.y << ")\n";
    std::cout << "Grid : (" << grid.x  << ", " << grid.y  << ")\n\n";

    // ======================================================================
    // Example from the lab PDF: 3×3 input, 2×2 kernel → 2×2 output
    // ======================================================================
    {
        std::cout << "--- Lab example (3×3 input, 2×2 kernel) ---\n";
        float exIn[9]  = {1,2,3, 4,5,6, 7,8,9};
        float exK[4]   = {0,1, 1,0};
        float exOut[4] = {0};

        // Upload to constant memory temporarily
        CUDA_CHECK(hipMemcpyToSymbol(c_kernel, exK, 4 * sizeof(float)));

        float *dExIn = nullptr, *dExOut = nullptr;
        CUDA_CHECK(hipMalloc(&dExIn,  9 * sizeof(float)));
        CUDA_CHECK(hipMalloc(&dExOut, 4 * sizeof(float)));
        CUDA_CHECK(hipMemcpy(dExIn, exIn, 9 * sizeof(float), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemset(dExOut, 0, 4 * sizeof(float)));

        dim3 bg(1,1), bb(2,2);
        conv2dTiled<<<bg, bb>>>(dExIn, dExOut, 3, 3, 2, 2, 2, 2);
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(exOut, dExOut, 4 * sizeof(float), hipMemcpyDeviceToHost));

        std::cout << "Expected output:\n  6  8\n 12 14\n";
        std::cout << "GPU output:\n";
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c)
                std::cout << "  " << (int)exOut[r * 2 + c];
            std::cout << "\n";
        }
        std::cout << "\n";

        CUDA_CHECK(hipFree(dExIn));
        CUDA_CHECK(hipFree(dExOut));

        // Restore the random kernel
        CUDA_CHECK(hipMemcpyToSymbol(c_kernel, hKernel.data(), szK * sizeof(float)));
    }

    // ======================================================================
    // Benchmark naive vs tiled on the large matrix
    // ======================================================================
    auto launchNaive = [&](){
        conv2dNaive<<<grid, block>>>(dIn, dOut,
                                    inRows, inCols, kRows, kCols,
                                    outRows, outCols);
    };
    auto launchTiled = [&](){
        conv2dTiled<<<grid, block>>>(dIn, dOut,
                                    inRows, inCols, kRows, kCols,
                                    outRows, outCols);
    };

    // --- Naive timing & result ---
    CUDA_CHECK(hipMemset(dOut, 0, szOut * sizeof(float)));
    float naiveMs = timeKernel(launchNaive);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpy(hOutNaive.data(), dOut, szOut * sizeof(float),
                          hipMemcpyDeviceToHost));

    // --- Tiled timing & result ---
    CUDA_CHECK(hipMemset(dOut, 0, szOut * sizeof(float)));
    float tiledMs = timeKernel(launchTiled);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpy(hOutTiled.data(), dOut, szOut * sizeof(float),
                          hipMemcpyDeviceToHost));

    // --- Print timing ---
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Timing (avg over 20 iterations) ===\n";
    std::cout << "  Naive (global mem) : " << naiveMs << " ms\n";
    std::cout << "  Tiled (shared mem) : " << tiledMs << " ms\n";
    std::cout << "  Speedup            : "
              << std::setprecision(2) << (naiveMs / tiledMs) << "x\n\n";

    // --- Correctness check against CPU (only for reasonably small sizes) ---
    if (szOut <= 1024UL * 1024UL) {
        std::cout << "Verifying against CPU reference...\n";
        conv2dCPU(hIn, hKernel, hRef, inRows, inCols, kRows, kCols, outRows, outCols);

        float maxErrNaive = 0.0f, maxErrTiled = 0.0f;
        for (size_t i = 0; i < szOut; ++i) {
            maxErrNaive = std::max(maxErrNaive, std::fabs(hOutNaive[i] - hRef[i]));
            maxErrTiled = std::max(maxErrTiled, std::fabs(hOutTiled[i] - hRef[i]));
        }
        std::cout << "  Naive max error: " << maxErrNaive << "\n";
        std::cout << "  Tiled max error: " << maxErrTiled << "\n";
        if (maxErrNaive < 1e-3f && maxErrTiled < 1e-3f)
            std::cout << "  PASS – both kernels match CPU reference.\n\n";
        else
            std::cout << "  FAIL – mismatch detected!\n\n";
    } else {
        std::cout << "(Skipping CPU verification for large output – too slow)\n\n";
    }

    // --- Clean up ---
    CUDA_CHECK(hipFree(dIn));
    CUDA_CHECK(hipFree(dOut));

    std::cout << "=== Done ===\n";
    return 0;
}
