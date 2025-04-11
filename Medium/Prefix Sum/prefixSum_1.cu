#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK = 256;
const int COARSE = 8;

__global__ void prefix_sum_kernel(const float* X, float* Y, int N, float* sum = nullptr) {
    __shared__ float X_s[BLOCK * COARSE];
    __shared__ float XY[BLOCK];
    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x * COARSE;
    // phase 1
    for(int i = 0 ; i < COARSE ; ++ i) {
        int pos = offset + tid + i * BLOCK;
        if(pos < N) X_s[tid + i * BLOCK] = X[pos];
        else X_s[tid + i * BLOCK] = 0;
    }
    __syncthreads();
    for(int i = 1 ; i < COARSE ; ++ i) {
        X_s[tid * COARSE + i] += X_s[tid * COARSE + i - 1];
    }
    __syncthreads();

    // phase 2
    XY[tid] = X_s[(tid + 1) * COARSE - 1];
    for(int stride = 1 ; stride < BLOCK ; stride *= 2) { 
        __syncthreads();  
        float temp;
        if(tid >= stride) {
            temp = XY[tid] + XY[tid - stride];
        }
        __syncthreads();
        if(tid >= stride)
            XY[tid] = temp;
    }

    // phase 3
    X_s[(tid + 1) * COARSE - 1] = XY[tid];
    for(int i = 0 ; i < COARSE - 1 ; ++ i) {
        X_s[(tid + 1) * COARSE + i] += XY[tid];
    }
    __syncthreads();

    if(sum != nullptr && tid == BLOCK - 1) {
        sum[blockIdx.x] = X_s[BLOCK * COARSE - 1];
    }
    for(int i = 0 ; i < COARSE ; ++ i) {
        int pos = offset + tid + i * BLOCK;
        if(pos < N) 
            Y[pos] = X_s[tid + i * BLOCK];
    }
}

__global__ void add_sum_kernel(float *output, float *sum, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = (bid + 1) * blockDim.x * COARSE;
    __shared__ float temp;
    if(tid == 0) temp = sum[bid];
    __syncthreads();
    for(int i = 0 ; i < COARSE ; ++ i) {
        int pos = offset + tid + i * BLOCK;
        if(pos < N) 
            output[pos] += temp;
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK;
    int numsPerBlock = BLOCK * COARSE;
    int blocksPerGrid = (N + numsPerBlock - 1) / numsPerBlock;
    float *sum;
    cudaMalloc((void**)&sum, sizeof(float) * blocksPerGrid);
    cudaMemset(sum, 0, sizeof(float) * blocksPerGrid);
    prefix_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, sum);
    cudaDeviceSynchronize();

    if(blocksPerGrid > 1) {
        prefix_sum_kernel<<<(blocksPerGrid + numsPerBlock - 1) / numsPerBlock, threadsPerBlock>>> (sum, sum, blocksPerGrid);
        cudaDeviceSynchronize();
        add_sum_kernel<<<blocksPerGrid - 1, threadsPerBlock>>>(output, sum, N);
        cudaDeviceSynchronize();
    }
}