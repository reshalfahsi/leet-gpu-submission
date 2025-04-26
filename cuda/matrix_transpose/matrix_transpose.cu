#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
/*
    Note:
    - For educational or prototype code, block size of 16 is a practical, “rule-of-thumb” choice often used in 
      CUDA tutorials and sample code for 2D problems.
    - CUDA GPUs execute threads in groups called warps (typically 32 threads). A block size of 256 is a multiple of 32, 
      ensuring full warps and avoiding wasted threads within a warp.
    - The number of threads per block impacts occupancy, the ratio of active warps to the maximum warps a GPU’s 
      streaming multiprocessor (SM) can handle. A block size of 256 often achieves good occupancy on many NVIDIA GPUs, 
      as it allows multiple blocks to run concurrently on an SM without exceeding resource limits (e.g., registers, shared memory).
    - Most NVIDIA GPUs have a maximum of 1024 threads per block (some newer architectures allow more, but 1024 is a safe limit for portability). 
      A 16 × 16 = 256 block is well below this limit, leaving room for flexibility and ensuring compatibility across GPU architectures.
*/


/* Naive solution */
/*
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // column index of input, row index of output
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row index of input, column index of output

    if (i < cols && j < rows) {
        output[i * rows + j] = input[j * cols + i];
    }
}
*/


/* Optimized with the tiling strategy, using shared memory to handle both reads and writes efficiently. */
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; // column index of input
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y; // row index of input

    // Read input into shared memory (coalesced)
    if (i < cols && j < rows) {
        tile[threadIdx.y][threadIdx.x] = input[j * cols + i];
    }
    __syncthreads();

    // Write from shared memory to output (transposed, coalesced)
    int out_i = blockIdx.y * BLOCK_SIZE + threadIdx.x; // row index of output
    int out_j = blockIdx.x * BLOCK_SIZE + threadIdx.y; // column index of output
    if (out_i < rows && out_j < cols) {
        output[out_j * rows + out_i] = tile[threadIdx.x][threadIdx.y];
    }
}


void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
