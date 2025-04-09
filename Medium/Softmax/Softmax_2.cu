#include "solve.h"
#include <cuda_runtime.h>
#include<iostream>

const int WARP_SIZE = 32;
const int BLOCK = 256;

//先在warp内执行规约操作
template<int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum)
{
    if(blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if(blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if(blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if(blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if(blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1); 
    return sum;
}

template<int blockSize>
__global__ void reduction_sum(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0;
    if(idx < N) sum += input[idx];
    if(idx + blockSize < N) sum += input[idx + blockSize];

    // shared mem for partial sums(one per warp in the block
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;

    // Final reduce using first warp
    if(warpId == 0)sum = warpReduceSum<blockSize / WARP_SIZE>(sum);

    if(tid == 0) 
        atomicAdd(output, sum);
}



template<int blockSize>
__device__ __forceinline__ float warpReduceMax(float maxv)
{
    if(blockSize >= 32) maxv = max(maxv, __shfl_down_sync(0xffffffff, maxv, 16));
    if(blockSize >= 16) maxv = max(maxv, __shfl_down_sync(0xffffffff, maxv, 8));
    if(blockSize >= 8) maxv = max(maxv, __shfl_down_sync(0xffffffff, maxv, 4));
    if(blockSize >= 4) maxv = max(maxv, __shfl_down_sync(0xffffffff, maxv, 2));
    if(blockSize >= 2) maxv = max(maxv, __shfl_down_sync(0xffffffff, maxv, 1));
    return maxv;
}

__device__ float atomicMax(float * addr, float val)
{
    int * addr_int = reinterpret_cast<int*>(addr);
    int * val_int = reinterpret_cast<int*>(&val);
    atomicMax(addr_int, *val_int);
}

template<int blockSize>
__global__ void reduction_max(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float maxv = 0;
    if(idx < N) maxv = max(maxv, input[idx]);
    if(idx + blockSize < N) maxv = max(maxv, input[idx + blockSize]);

    static __shared__ float warpLevelMaxv[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    maxv = warpReduceMax<blockSize>(maxv);

    if(laneId == 0) warpLevelMaxv[warpId] = maxv;
    __syncthreads();

    maxv = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxv[laneId] : 0;

    // Final reduce using first warp
    if(warpId == 0) maxv = warpReduceMax<blockSize / WARP_SIZE>(maxv);

    if(tid == 0) 
        atomicMax(output, maxv);
}

__global__ void exp_kernel(const float* input, const float* maxv, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N) {
        output[idx] = expf(input[idx] - (*maxv));
    }
}

__global__ void softmax_kernel(float* input, const float* sum, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N) {
        input[idx] = input[idx] / (*sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    float *maxv;
    float *sum;
    cudaMalloc((void**)&maxv, sizeof(float));
    cudaMalloc((void**)&sum, sizeof(float)); 
    cudaMemset(maxv, 0, sizeof(float));
    cudaMemset(sum, 0, sizeof(float));

    int threadsPerBlock = BLOCK;
    int blocks = BLOCK * 2;
    int blockPerGrid = (N + blocks - 1) / blocks;
    reduction_max<BLOCK><<<blockPerGrid, threadsPerBlock>>>(input, maxv, N);
    cudaDeviceSynchronize();
    exp_kernel<<<BLOCK, (N + BLOCK - 1) / BLOCK>>>(input, maxv, output, N);
    cudaDeviceSynchronize();
    reduction_sum<BLOCK><<<blockPerGrid, threadsPerBlock>>>(output, sum, N);
    cudaDeviceSynchronize();
    softmax_kernel<<<BLOCK, (N + BLOCK - 1) / BLOCK>>>(output, sum, N);
    cudaDeviceSynchronize();
}