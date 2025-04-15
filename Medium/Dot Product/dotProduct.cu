#include "solve.h"
#include <cuda_runtime.h>
const int WARP_SIZE = 32;
const int BLOCK = 256;

template<int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if(blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if(blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if(blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if(blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if(blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1); 
    return sum;    
}

template<int blockSize>
__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0;
    if(idx < N) sum = A[idx] * B[idx];
    if(idx + BLOCK < N) sum += A[idx + BLOCK] * B[idx + BLOCK];

    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);
    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if(warpId == 0)sum = warpReduceSum<blockSize / WARP_SIZE>(sum);

    if(tid == 0) 
        atomicAdd(result, sum);    
}

// A, B, result are device pointers
void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = BLOCK;
    int numsPerBlock = BLOCK * 2;
    int blocksPerGrid = (N + numsPerBlock - 1) / numsPerBlock;
    dot_product_kernel<BLOCK><<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}