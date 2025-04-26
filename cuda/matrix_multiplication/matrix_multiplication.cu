#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    /*
        Each thread computes one element C[i][j] of the output matrix.

        Note:
        - In the CUDA program, all matrices (A, B, and C) are stored in row-major format as one-dimensional arrays in memory.
        - The “two-dimensional” description refers to its logical matrix structure, not its memory layout.
    */

    /*
        - blockIdx.y: Block index along the y-axis (row direction).
        - blockDim.y: Number of threads per block in the y-direction.
        - threadIdx.y: Thread’s local index within the block in the y-direction.
    */
    int i = blockIdx.y * blockDim.y + threadIdx.y; // rows of C

    /*
        - blockIdx.x: Block index along the x-axis (columns direction).
        - blockDim.x: Number of threads per block in the x-direction.
        - threadIdx.x: Thread’s local index within the block in the x-direction. 
    */     
    int j = blockIdx.x * blockDim.x + threadIdx.x; // cols of C

    /* 
        Ensures the thread only processes valid indices. 
        Since the grid may launch more threads than needed (due to ceiling division), this prevents out-of-bounds memory access. 
    */
    if (i < M && j < K) { 
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * K + j];
        }
        C[i * K + j] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    /*
        This function configures and launches the CUDA kernel on the GPU.

        Parameters:
        - const float* A: Device pointer to matrix A (size M × N).
        - const float* B: Device pointer to matrix B (size N × K).
        - float* C: Device pointer to matrix C (size M × K).
        - int M, N, K: Dimensions of the matrices (A is M × N, B is N × K, C is M × K).
    */

    // dim3 is a CUDA type for specifying dimensions (x, y, z), here used for 2D blocks.
    dim3 threadsPerBlock(16, 16); // Each block contains a 2D grid of 16 × 16 = 256 threads.

    /*
        - blocksPerGrid is computed to ensure enough blocks cover the entire output matrix C (size M × K).
        - blocksPerGrid.x = (K + threadsPerBlock.x - 1) / threadsPerBlock.x: Number of blocks along the x-axis, covering K columns of C.
        - blocksPerGrid.y = (M + threadsPerBlock.y - 1) / threadsPerBlock.y: Number of blocks along the y-axis, covering M rows of C.
    */
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launches the kernel with the specified grid and block dimensions.
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    /* 
        Ensures the kernel completes before the function returns. 
        Since kernel launches are asynchronous, this is necessary to guarantee C contains the final results.
    */
    cudaDeviceSynchronize();
}
