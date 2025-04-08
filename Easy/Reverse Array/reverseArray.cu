#include "solve.h"
#include <cuda_runtime.h>
const int BLOCK_SIZE = 256;

// 使用 shared memory 来做 in-place 翻转
__global__ void reverse_array(float* input, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    __shared__ float pre[BLOCK_SIZE];
    __shared__ float suf[BLOCK_SIZE];
    if(idx < (N + 1) / 2) {
        pre[tid] = input[idx];
        suf[tid] = input[N - idx - 1];
    }
    __syncthreads();
    if(idx < (N + 1) / 2) {
        input[idx] = suf[tid];
        input[N - idx - 1] = pre[tid];
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocks = BLOCK_SIZE * 2;
    int blocksPerGrid = (N + blocks - 1) / blocks;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}