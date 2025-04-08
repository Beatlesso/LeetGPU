#include "solve.h"
#include <cuda_runtime.h>

const int MAXN = 1024;

__device__ float atomicMax(float * addr, float val)
{
    int * addr_int = reinterpret_cast<int*>(addr);
    int * val_int = reinterpret_cast<int*>(&val);
    atomicMax(addr_int, *val_int);
}

// 用了 shared memory, 但是是限定在一个 block 内完成的 naive 版本
__global__ void softmax_kernel(const float* input, float* output, int N) {
    // TODO: Implement the softmax kernel.  Remember to use the max trick!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sh_sum[MAXN];
    __shared__ float sh_maxv[MAXN];
    sh_sum[idx] = 0;
    sh_maxv[idx] = 0;
    __syncthreads();
    if(idx < blockDim.x) {
        for(int i = idx ; i < N ; i += blockDim.x) {
            sh_maxv[idx] = max(sh_maxv[idx], input[i]);
        }
    }
    __syncthreads();
    atomicMax(&sh_maxv[0], sh_maxv[idx]);
    __syncthreads();
    if(idx < blockDim.x) {
        for(int i = idx ; i < N ; i += blockDim.x) {
            float ep = expf((input[i] - sh_maxv[0]));
            sh_sum[idx] += ep;
            output[i] = ep;
        }
    }
    __syncthreads();
    if(idx != 0) atomicAdd(&sh_sum[0], sh_sum[idx]);
    __syncthreads();
    if(idx < blockDim.x) {
        for(int i = idx ; i < N ; i += blockDim.x) {
            output[i] /= sh_sum[0];
        }
    }
    __syncthreads();
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = MAXN;
    int blocksPerGrid = 1;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}