#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK = 256;
const int COARSE = 4;

__global__ void prefix_sum_kernel(const float* X, float* Y, int N, float* sum = nullptr) {
    __shared__ float X_s[BLOCK * COARSE];
    // double buffer
    __shared__ float XY0[BLOCK];
    __shared__ float XY1[BLOCK];
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
    XY0[tid] = X_s[(tid + 1) * COARSE - 1];
    __syncthreads();
    int t = 1;
    for(int stride = 1 ; stride < BLOCK ; stride *= 2) { 
        if(t & 1) {
            if(tid >= stride) {
                XY1[tid] = XY0[tid] + XY0[tid - stride];
            } else {
                XY1[tid] = XY0[tid];
            }
        }
        else {
            if(tid >= stride) {
                XY0[tid] = XY1[tid] + XY1[tid - stride];
            } else {
                XY0[tid] = XY1[tid];
            }            
        }
        ++ t;
        __syncthreads();
    }

    // phase 3
    if(t & 1) {
        X_s[(tid + 1) * COARSE - 1] = XY0[tid];
        for(int i = 0 ; i < COARSE - 1 ; ++ i) {
            X_s[(tid + 1) * COARSE + i] += XY0[tid];
        }
    } else {
        X_s[(tid + 1) * COARSE - 1] = XY1[tid];
        for(int i = 0 ; i < COARSE - 1 ; ++ i) {
            X_s[(tid + 1) * COARSE + i] += XY1[tid];
        }
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