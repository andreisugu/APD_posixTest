// matrix_multiply.cu
//
// Lab 1 – Assignment 1
// Multiplies two matrices A (M×N) and B (N×K) on the GPU
// producing C (M×K) = A × B.
//
// Compile:
//   nvcc -O3 matrix_multiply.cu -o matrix_multiply
// Run:
//   ./matrix_multiply          (uses default 512x512 x 512x512)
//   ./matrix_multiply 256 256 256

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

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------
#define TILE_SIZE 16   // threads per block dimension (16×16 = 256 threads)

// --------------------------------------------------------------------------
// Tiled matrix-multiply kernel
//
// Each thread block loads a TILE_SIZE×TILE_SIZE sub-tile of A and B into
// shared memory, accumulates the partial dot products, then advances to the
// next tile along the shared K dimension.  This maximises data re-use and
// gives coalesced global-memory accesses in both A and B.
//
// A: row-major, dimensions M×N  (device pointer)
// B: row-major, dimensions N×K  (device pointer)
// C: row-major, dimensions M×K  (device pointer)
// --------------------------------------------------------------------------
__global__ void matMulTiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float*       __restrict__ C,
                             int M, int N, int K)
{
    // Shared-memory tiles for the current A and B sub-blocks
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Global row / column that this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;   // row in C / A
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;   // col in C / B

    float sum = 0.0f;

    // Loop over every tile along the K-dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {

        // -------------------------------------------------------------------
        // Collaboratively load one TILE_SIZE×TILE_SIZE block of A into shared
        // memory.  A[row][t*TILE_SIZE + threadIdx.x]
        // -------------------------------------------------------------------
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // -------------------------------------------------------------------
        // Collaboratively load one TILE_SIZE×TILE_SIZE block of B into shared
        // memory.  B[t*TILE_SIZE + threadIdx.y][col]
        // -------------------------------------------------------------------
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < N && col < K)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // All threads in the block must finish loading before any thread
        // starts computing with the shared data.
        __syncthreads();

        // Accumulate the dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // Prevent any thread from overwriting the shared tiles before every
        // thread has finished reading them.
        __syncthreads();
    }

    // Write the final result
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// --------------------------------------------------------------------------
// CPU reference implementation (for correctness verification)
// --------------------------------------------------------------------------
static void matMulCPU(const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>&       C,
                      int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * K + j];
            C[i * K + j] = s;
        }
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Matrix dimensions – override from command line if desired
    int M = 512, N = 512, K = 512;
    if (argc == 4) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::cout << "=== CUDA Matrix Multiply  A(" << M << "x" << N << ") * B("
              << N << "x" << K << ") = C(" << M << "x" << K << ") ===\n\n";

    // ------------------------------------------------------------------
    // Host data
    // ------------------------------------------------------------------
    const size_t szA = (size_t)M * N;
    const size_t szB = (size_t)N * K;
    const size_t szC = (size_t)M * K;

    std::vector<float> hA(szA), hB(szB), hC(szC, 0.0f), hRef(szC, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    // ------------------------------------------------------------------
    // Device memory
    // ------------------------------------------------------------------
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(hipMalloc(&dA, szA * sizeof(float)));
    CUDA_CHECK(hipMalloc(&dB, szB * sizeof(float)));
    CUDA_CHECK(hipMalloc(&dC, szC * sizeof(float)));

    CUDA_CHECK(hipMemcpy(dA, hA.data(), szA * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dB, hB.data(), szB * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(dC, 0, szC * sizeof(float)));

    // ------------------------------------------------------------------
    // Launch configuration
    // ------------------------------------------------------------------
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Block: (" << block.x << ", " << block.y << ")\n";
    std::cout << "Grid : (" << grid.x  << ", " << grid.y  << ")\n\n";

    // ------------------------------------------------------------------
    // Timing with CUDA events
    // ------------------------------------------------------------------
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));

    // Warm-up run
    matMulTiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(hipDeviceSynchronize());

    CUDA_CHECK(hipEventRecord(start));
    matMulTiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(hipEventRecord(stop));
    CUDA_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(hipGetLastError());

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU kernel time : " << ms << " ms\n\n";

    // ------------------------------------------------------------------
    // Copy result back and verify (only for small matrices)
    // ------------------------------------------------------------------
    CUDA_CHECK(hipMemcpy(hC.data(), dC, szC * sizeof(float), hipMemcpyDeviceToHost));

    if (M <= 256 && N <= 256 && K <= 256) {
        std::cout << "Verifying result against CPU reference...\n";
        matMulCPU(hA, hB, hRef, M, N, K);

        float maxErr = 0.0f;
        for (size_t i = 0; i < szC; ++i)
            maxErr = std::max(maxErr, std::fabs(hC[i] - hRef[i]));

        std::cout << "Max absolute error: " << maxErr << "\n";
        if (maxErr < 1e-2f)
            std::cout << "PASS – results match CPU reference.\n\n";
        else
            std::cout << "FAIL – results differ from CPU reference!\n\n";
    } else {
        std::cout << "(Skipping CPU verification for large matrix – too slow)\n\n";
    }

    // ------------------------------------------------------------------
    // Print a small corner of the result
    // ------------------------------------------------------------------
    int printRows = std::min(M, 4), printCols = std::min(K, 4);
    std::cout << "Top-left " << printRows << "x" << printCols << " of C:\n";
    for (int i = 0; i < printRows; ++i) {
        for (int j = 0; j < printCols; ++j)
            std::cout << std::setw(10) << hC[i * K + j] << " ";
        std::cout << "\n";
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    CUDA_CHECK(hipFree(dA));
    CUDA_CHECK(hipFree(dB));
    CUDA_CHECK(hipFree(dC));
    CUDA_CHECK(hipEventDestroy(start));
    CUDA_CHECK(hipEventDestroy(stop));

    std::cout << "\n=== Done ===\n";
    return 0;
}
