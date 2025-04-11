#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK = 256;

// 合并访存，减少控制发散
__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float input_s[BLOCK];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    if(idx < N) input_s[tid] = input[idx];
    else input_s[tid] = 0;
    if(idx + BLOCK < N) input_s[tid] += input[idx + BLOCK];
    for(int stride = BLOCK / 2 ; stride >= 1 ; stride /= 2) {
        __syncthreads();
        if(tid < stride) 
            input_s[tid] += input_s[tid + stride];
    }
    if(tid == 0) 
        atomicAdd(output, input_s[0]);
}


// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = BLOCK;
    int blocks = BLOCK * 2;
    int blockPerGrid = (N + blocks - 1) / blocks;
    reduction_kernel<<<blockPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}