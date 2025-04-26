#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    /*
        This kernel performs the element-wise addition of two input vectors A and B, storing the results in C.

        Parameters:
        - const float* A, const float* B: Pointers to the input vectors (read-only, hence const).
        - float* C: Pointer to the output vector (writable).
        - int N: The length of the vectors.
    */

    /*
        This computes a unique global index for each thread.
        - blockIdx.x is the block ``index``, 
        - blockDim.x is the ``number`` of threads per block
        - threadIdx.x is the thread’s ``index`` within its block.

        Note: An ``index`` is moving, a ``number`` is constant
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    /*
        Threads with idx >= N do nothing, preventing out-of-bounds access.
        Example:
            For N = 1000, 1280 threads are launched, so indices 1000 to 1279 are out of bounds.
    */
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    /*
        Launches the kernel on the GPU.

        Parameters:
        - const float* A, const float* B: Pointers to the input vectors (read-only, hence const).
        - float* C: Pointer to the output vector (writable).
        - int N: The length of the vectors.
    */

    int threadsPerBlock = 256; // Number of threads per block.
    
    /*
        Calculates the number of blocks needed, using ceiling division to ensure all elements are processed.
    */
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N); // dispatches the kernel.

    cudaDeviceSynchronize(); // ensures the kernel completes before the function returns.
}
