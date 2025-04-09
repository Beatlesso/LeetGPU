#include "solve.h"
#include <cuda_runtime.h>
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
__global__ void reduction_kernel(const float* input, float* output, int N) {
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


// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = BLOCK;
    int blocks = BLOCK * 2;
    int blockPerGrid = (N + blocks - 1) / blocks;
    reduction_kernel<BLOCK><<<blockPerGrid, threadsPerBlock>>>(input, output, N);
}