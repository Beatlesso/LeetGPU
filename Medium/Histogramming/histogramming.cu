#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK = 256;
const int COSER = 8;

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int shared_mem[];
    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x * COSER;
    for(int i = tid ; i < num_bins ; i += BLOCK) 
        shared_mem[i] = 0;
    __syncthreads();
    for(int i = 0 ; i < COSER ; ++ i) {
        int idx = tid + offset + i * BLOCK;
        if(idx >= N) break;
        atomicAdd(&shared_mem[input[idx]], 1);
    }
    __syncthreads();
    for(int i = tid ; i < num_bins ; i += BLOCK)  
        atomicAdd(&histogram[tid], shared_mem[tid]);
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = BLOCK;
    int numPerBlock = BLOCK * COSER;
    int grid = (N + numPerBlock - 1) / numPerBlock;
    int shared_size = num_bins * sizeof(int);
    histogram_kernel<<<grid, threadsPerBlock, shared_size>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
