# CUDA Lab 1 & Lab 2 – Complete Solutions Guide

> **Note:** These files have been converted to **HIP** for AMD iGPU compatibility. All CUDA API calls (`cuda*`) have been replaced with HIP equivalents (`hip*`). See [Compile & Run Commands](#compile--run-commands) section for both AMD (HIP) and NVIDIA (CUDA) compilation options.

## Files Provided

| File | Lab | Assignment |
|------|-----|-----------|
| `vector_add.cu` | Lab 1 | Given – vector addition starter |
| `matrix_transpose_shared.cu` | Lab 2 | Given – naive + tiled transpose |
| `matrix_multiply.cu` | Lab 1 | **Assignment 1** – Matrix multiplication |
| `matrix_transpose.cu` | Lab 1 | **Assignment 2** – Matrix transpose (global memory) |
| `conv2d_shared.cu` | Lab 2 | **Assignment 1** – 2D Convolution (naive + tiled) |

---

## Compile & Run Commands

### For AMD iGPU (HIP)

```bash
# Lab 1 – Matrix Multiply
hipcc -O3 matrix_multiply.cu -o matrix_multiply
./matrix_multiply              # default 512×512 × 512×512
./matrix_multiply 256 256 256  # M=256, N=256, K=256

# Lab 1 – Matrix Transpose
hipcc -O3 matrix_transpose.cu -o matrix_transpose
./matrix_transpose             # default 1024×1024
./matrix_transpose 2048 2048

# Lab 2 – 2D Convolution
hipcc -O3 conv2d_shared.cu -o conv2d_shared
./conv2d_shared                # default 1024×1024 input, 5×5 kernel
./conv2d_shared 2048 2048 7 7  # 2048×2048 input, 7×7 kernel
```

### For NVIDIA GPU (Original CUDA)

```bash
# Lab 1 – Matrix Multiply
nvcc -O3 matrix_multiply.cu -o matrix_multiply
./matrix_multiply              # default 512×512 × 512×512
./matrix_multiply 256 256 256  # M=256, N=256, K=256

# Lab 1 – Matrix Transpose
nvcc -O3 matrix_transpose.cu -o matrix_transpose
./matrix_transpose             # default 1024×1024
./matrix_transpose 2048 2048

# Lab 2 – 2D Convolution
nvcc -O3 conv2d_shared.cu -o conv2d_shared
./conv2d_shared                # default 1024×1024 input, 5×5 kernel
./conv2d_shared 2048 2048 7 7  # 2048×2048 input, 7×7 kernel
```

---

## Lab 1 – Assignment 1: Matrix Multiplication

### Algorithm

Compute **C = A × B** where A is M×N and B is N×K, producing C of M×K.

```
C[i][j] = sum_{k=0}^{N-1} A[i][k] * B[k][j]
```

### Key CUDA Concepts Used

**Tiled (shared memory) approach:**

Each thread block processes a `TILE_SIZE × TILE_SIZE` (16×16) sub-block of the output
matrix C. To compute the values in that sub-block, threads cooperate to load matching
tiles of A and B into shared memory, one tile-column-of-A / tile-row-of-B at a time.

```
for t = 0 to numTiles-1:
    tileA[ty][tx] = A[row][t*TILE + tx]   // load from global → shared
    tileB[ty][tx] = B[t*TILE + ty][col]   // load from global → shared
    __syncthreads()                         // wait for all loads
    sum += tileA[ty][k] * tileB[k][tx]    // accumulate using shared
    __syncthreads()                         // wait before overwriting tiles
```

**Why tiling is faster:**
- Each element of A and B is loaded from global memory once per block column/row,
  then reused TILE_SIZE times from fast shared memory.
- Memory bandwidth saving factor ≈ TILE_SIZE (16×).

### Thread Index Mapping

```
int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // row in C
int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // col in C
```

---

## Lab 1 – Assignment 2: Matrix Transpose (Global Memory)

### Algorithm

Given A of dimensions `rows × cols`, produce A^T of dimensions `cols × rows`:

```
A^T[j][i] = A[i][j]
```

### Implementation

- Block: `(TILE_DIM=32, BLOCK_ROWS=8)` → 256 threads per block
- Each block handles a 32×32 tile of the matrix
- Each thread handles 4 rows (TILE_DIM / BLOCK_ROWS = 4)

```cpp
int x = blockIdx.x * TILE_DIM + threadIdx.x;  // column in input
int y = blockIdx.y * TILE_DIM + threadIdx.y;  // row in input

// Global reads are coalesced (consecutive threads → consecutive columns)
// Global writes are strided (write to out[col][row] = out[x][y])
out[x * height + yy] = in[yy * width + x];
```

**Note:** This is the same naive kernel as in `matrix_transpose_shared.cu`.
Lab 2 adds the tiled version to fix the non-coalesced writes.

---

## Lab 2 – Assignment 1: 2D Convolution

### Algorithm (valid convolution)

```
output[i][j] = sum_{m=0}^{kRows-1}  sum_{n=0}^{kCols-1}
               input[i+m][j+n] * kernel[m][n]

output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1
```

### Lab Example Verification

```
Input (3×3):          Kernel (2×2):    Output (2×2):
  1 2 3                 0 1              6  8
  4 5 6     *           1 0     =       12 14
  7 8 9

output[0][0] = 1*0 + 2*1 + 4*1 + 5*0 = 6   ✓
output[0][1] = 2*0 + 3*1 + 5*1 + 6*0 = 8   ✓
output[1][0] = 4*0 + 5*1 + 7*1 + 8*0 = 12  ✓
output[1][1] = 5*0 + 6*1 + 8*1 + 9*0 = 14  ✓
```

### Naive Kernel

Each output thread independently loops over the entire kernel, reading each
required input value directly from **global memory**.

```cpp
for (int m = 0; m < kRows; m++)
    for (int n = 0; n < kCols; n++)
        sum += input[(outRow+m)*inCols + (outCol+n)] * c_kernel[m*kCols+n];
```

**Problem:** For a 5×5 kernel, every output thread reads 25 global memory
locations. Adjacent threads re-read many of the same input values from global
memory independently — very wasteful.

### Tiled (Shared Memory) Kernel

Each block cooperatively loads a **halo tile** of the input into shared memory:

```
Shared tile size = (BLOCK_H + kRows - 1) × (BLOCK_W + kCols - 1)
```

For a 16×16 output block with a 5×5 kernel:
```
Shared tile = (16+4) × (16+4) = 20×20 = 400 floats = 1600 bytes
```

Loading process:
1. All 256 threads cooperate to fill the shared tile (each loads ~1–2 elements)
2. `__syncthreads()` ensures all data is ready
3. Each thread reads from fast shared memory instead of slow global memory

**The kernel filter** is stored in `__constant__` memory, which is cached and
broadcast efficiently to all threads reading the same filter weight.

### Why Constant Memory for the Kernel?

All threads in a warp read the **same** filter value `c_kernel[m][n]` at the
same time — this is a broadcast pattern that constant memory is optimized for.
It avoids hitting global memory for filter reads entirely.

### Performance Analysis

| Kernel size | Global reads (naive) | Global reads (tiled, per thread) |
|-------------|---------------------|----------------------------------|
| 3×3         | 9                   | ~1 (shared amortized)            |
| 5×5         | 25                  | ~1.5625                          |
| 7×7         | 49                  | ~1.89                            |

The tiled version amortizes global reads across the block — each input value
is loaded once and reused by multiple threads.

---

## CUDA Memory Hierarchy Summary

```
Fastest  ┌──────────────────────────────────────────┐
         │  Registers    (per thread, ~255 regs)    │
         ├──────────────────────────────────────────┤
         │  Shared Memory (per block, ~48 KB)       │  ← key to performance
         ├──────────────────────────────────────────┤
         │  Constant Memory (read-only, cached)     │  ← good for filter weights
         ├──────────────────────────────────────────┤
         │  L2 Cache                                │
         ├──────────────────────────────────────────┤
         │  Global Memory (GB range, ~600 GB/s)     │  ← must minimize accesses
Slowest  └──────────────────────────────────────────┘
```

---

## Key CUDA Patterns in All Solutions

### 1. CUDA/HIP Error Checking

**CUDA version:**
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { ... std::exit(1); } \
} while(0)
```

**HIP version (AMD iGPU):**
```cpp
#define CUDA_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { ... std::exit(1); } \
} while(0)
```

Always check return values from every CUDA/HIP API call.

### 2. Kernel Timing with CUDA/HIP Events

**CUDA version:**
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
myKernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

**HIP version (AMD iGPU):**
```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);
hipEventRecord(start);
hipLaunchKernelGGL(myKernel, grid, block, 0, 0, ...);
hipEventRecord(stop);
hipEventSynchronize(stop);
float ms;
hipEventElapsedTime(&ms, start, stop);
```

**Note:** HIP uses `hipLaunchKernelGGL()` for kernel launches instead of `<<<>>>` syntax.

### 3. Shared Memory + Synchronization Pattern

```cpp
__shared__ float tile[...];
// Phase 1: cooperative load
tile[threadIdx.y][threadIdx.x] = input[...];
__syncthreads();           // barrier – all threads must finish loading
// Phase 2: compute using tile
float sum = tile[...][...] * ...;
__syncthreads();           // barrier before next iteration overwrites tile
```

**Compatible with both CUDA and HIP** – the `__shared__` and `__syncthreads()` syntax is the same.

### 4. Boundary Guards
Always check bounds before reading/writing:
```cpp
if (row < rows && col < cols)
    output[row * cols + col] = result;
```
Without this, threads at the edge of the grid write to invalid memory
when matrix dimensions aren't exact multiples of block size.
