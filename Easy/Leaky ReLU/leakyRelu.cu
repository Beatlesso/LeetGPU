#include "solve.h"
#include <cuda_runtime.h>

const float alpha = 0.01;
const int COSER = 2;
const int BLOCK = 256;
// 线程粗化可以 90% percentile
__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x * COSER;
    for(int i = 0 ; i < COSER ; ++ i) {
        int pos = idx + i * BLOCK;
        if(pos < N) {
            float x = input[pos];
            output[pos] = x > 0 ? x : x * alpha;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK;
    int blocks = BLOCK * COSER;
    int blocksPerGrid = (N + blocks - 1) / blocks;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}