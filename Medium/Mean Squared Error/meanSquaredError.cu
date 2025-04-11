#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK = 256;
const int COSER = 2;
const int WARP_SIZE = 32;

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
__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int N) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * COSER * blockIdx.x;
    float sum = 0;
    if(idx < N) {
        float x = predictions[idx] - targets[idx];
        sum += x * x / N;
    }
    if(idx + BLOCK < N) {
        float x = predictions[idx + BLOCK] - targets[idx + BLOCK];
        sum += x * x / N;
    }
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int wrapId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);
    if(laneId == 0) warpLevelSums[wrapId] = sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if(wrapId == 0) sum = warpReduceSum<blockSize / WARP_SIZE>(sum);

    if(tid == 0)
        atomicAdd(mse, sum);
}

// predictions, targets, mse are device pointers
void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsPerBlock = BLOCK;
    int numsPerBlock = BLOCK * COSER;
    int blocksPerGrid = (N + numsPerBlock - 1) / numsPerBlock;
    mse_kernel<BLOCK><<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
