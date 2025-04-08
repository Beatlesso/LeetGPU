#include "solve.h"
#include <cuda_runtime.h>
const int TileSize = 32;

// 使用了 Tiling 优化的矩阵乘法
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int x = threadIdx.x, y = threadIdx.y;
    int arow = blockIdx.x * blockDim.x + x;
    int acol = blockIdx.y * blockDim.y + y;
    __shared__ float TilingA[TileSize][TileSize];
    __shared__ float TilingB[TileSize][TileSize];
    if(arow < M && acol < N) {
        TilingA[x][y] = A[arow * N + acol];
    } else {
        TilingA[x][y] = 0;
    }
    __syncthreads();
    
    for(int i = 0 ; i < (K + TileSize - 1) / TileSize ; i ++) {
        int bcol = i * TileSize + y;
        int brow = blockIdx.y * blockDim.y + x;
        if(brow < N && bcol < K) {
            TilingB[x][y] = B[brow * K + bcol];
        } else {
            TilingB[x][y] = 0;
        }
        __syncthreads();
        // Kahan sum
        float sum = 0.0f;
        float c = 0.0f;
        for(int j = 0 ; j < TileSize ; j ++) {
            float num = TilingA[x][j] * TilingB[j][y];
            float a = num - c;
            float b = sum + a;
            c = (b - sum) - a;
            sum = b;
        }
        
        int crow = arow, ccol = bcol;
        if(crow < M && ccol < K)
            atomicAdd(&C[arow * K + bcol], sum);
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TileSize, TileSize);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
