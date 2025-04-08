#include "solve.h"
#include <cuda_runtime.h>
const int BLOCK_SIZE = 256;

// naive 实现，只考虑了让相邻线程访存尽可能连续
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int x = threadIdx.x;
    int offset = blockDim.x * blockIdx.x * 4;
    for(int i = 0 ; i < 4 ; ++ i) {
        int ptr = offset + x + i * BLOCK_SIZE;
        if(ptr < width * height * 4) {
            if(x % 4 != 3)
                image[ptr] = 255 - image[ptr];
            else
                image[ptr] = image[ptr];
        }
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}