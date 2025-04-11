#include <iostream>
#include <cuda_runtime.h>

/*
NVIDIA TESLA T4
max shared memory per block: 49152 bytes
SM count: 40
Max shared memory per SM: 65536 bytes    
*/

int main() {
    cudaSetDevice(0);
    int shared_mem_per_block;
    cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    std::cout << " max shared memory per block: " << shared_mem_per_block << " bytes" << std::endl;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    std::cout << " SM count: " << sm_count << std::endl;
    int max_shared_memory_per_sm;
    cudaDeviceGetAttribute(&max_shared_memory_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    std::cout << "Max shared memory per SM: " << max_shared_memory_per_sm << " bytes" << std::endl;
    return 0;
}
