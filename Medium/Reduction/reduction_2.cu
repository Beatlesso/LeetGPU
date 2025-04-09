#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK = 256;

// 展开最后一维减少同步
__device__ void wrapReduce(volatile float* cache, int tid) {
    if(tid < 32) cache[tid] += cache[tid+32];
    __syncwarp();
    if(tid < 16) cache[tid] += cache[tid+16];
    __syncwarp();
    if(tid < 8) cache[tid] += cache[tid+8];
    __syncwarp();
    if(tid < 4) cache[tid] += cache[tid+4];
    __syncwarp();
    if(tid < 2) cache[tid] += cache[tid+2];
    __syncwarp();
    if(tid < 1) cache[tid] += cache[tid+1];
}

__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float input_s[BLOCK];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    if(idx < N) input_s[tid] = input[idx];
    else input_s[tid] = 0;
    if(idx + BLOCK < N) input_s[tid] += input[idx + BLOCK];
    for(int stride = BLOCK / 2 ; stride > 32 ; stride /= 2) {
        __syncthreads();
        if(tid < stride) 
            input_s[tid] += input_s[tid + stride];
    }
    __syncthreads();
    if(tid < 32)
        wrapReduce(input_s, tid);
    if(tid == 0) 
        atomicAdd(output, input_s[0]);
}


// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = BLOCK;
    int blocks = BLOCK * 2;
    int blockPerGrid = (N + blocks - 1) / blocks;
    reduction_kernel<<<blockPerGrid, threadsPerBlock>>>(input, output, N);
}