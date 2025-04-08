#include "solve.h"
#include <cuda_runtime.h>
const int BLOCK_SIZE = 256;
const int COSER = 2;
// 线程粗化可以略微提高 percentile
__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x * COSER;
    for(int i = 0 ; i < COSER ; ++ i) {
        int pos = idx + i * BLOCK_SIZE;
        if(pos < N)
            output[pos] = max(0.0, input[pos]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocks = BLOCK_SIZE * COSER;
    int blocksPerGrid = (N + blocks - 1) / blocks;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}