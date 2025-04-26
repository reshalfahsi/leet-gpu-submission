#include "solve.h"
#include <cuda_runtime.h>

#define RGBA_SIZE 4

__global__ void invert_kernel(unsigned char* image, const int width, const int height) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < width * height){
        for (int color_index=0; color_index < (RGBA_SIZE - 1); color_index++){
            image[index * RGBA_SIZE + color_index] = 255 - image[index * RGBA_SIZE + color_index];
        }
    }

}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
