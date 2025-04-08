#include "solve.h"
#include <cuda_runtime.h>
const int BLOCK_SIZE = 256;

// 使用了 shared memory 进行优化
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    extern __shared__ float shared_mem[];
    int len_input_s = BLOCK_SIZE + kernel_size - 1;
    int len_output_s = BLOCK_SIZE;
    int total_len = len_input_s + len_output_s + kernel_size;
    float* input_s = shared_mem;
    float* output_s = shared_mem + BLOCK_SIZE + kernel_size - 1;
    float* kernel_s = output_s + BLOCK_SIZE;

    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x;
    int idx = tid + offset;
    int output_size = input_size - kernel_size + 1;

    for(int i = tid ; i < len_output_s ; i += BLOCK_SIZE)
        output_s[i] = 0;
    for(int i = tid ; i + offset < input_size && i < len_input_s ; i += BLOCK_SIZE)    
        input_s[i] = input[offset + i];
    for(int i = tid ; i < kernel_size ; i += BLOCK_SIZE)
        kernel_s[i] = kernel[i];
    __syncthreads();
    if(idx < output_size) {
        for(int i = 0 ; i < kernel_size ; ++ i)
            output_s[tid] += kernel_s[i] * input_s[tid + i];
        output[idx] = output_s[tid];        
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    int input_s_size = threadsPerBlock + kernel_size - 1;
    int output_s_size = threadsPerBlock;
    int shared_mem_size = (input_s_size + output_s_size + kernel_size) * sizeof(float);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}