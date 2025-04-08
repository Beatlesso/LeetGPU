#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK_SIZE = 16;
// 非常 naive 的版本, 访存不连续
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < rows && y < cols) {
        output[y * rows + x] = input[x * cols + y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (cols + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}