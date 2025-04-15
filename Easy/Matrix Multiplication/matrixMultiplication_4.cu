#include "solve.h"
#include <cuda_runtime.h>

const int TileSize = 32;
const int multiB = 8;

// A(M, N) B(N, K) C(M, K)
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float TileA[TileSize][TileSize];
    __shared__ float TileB[TileSize][TileSize * multiB];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum[multiB];
    for(int i = 0 ; i < multiB ; ++ i) sum[i] = 0;
    // 当前 C 中 TileBlock 的左上角 行列坐标
    int row = blockDim.y * blockIdx.y;
    int col = blockDim.x * blockIdx.x * multiB;
    for(int i = 0 ; i < (N + TileSize - 1) / TileSize ; ++ i) { // 枚举 TileA TileB
        int Arow = row + ty;
        int Acol = i * TileSize + tx;
        int Brow = i * TileSize + ty;
        int Bcol = col + tx;
        int Aptr = Arow * N + Acol;
        int Bptr = Brow * K + Bcol;

        if(Arow < M && Acol < N) TileA[ty][tx] = A[Aptr];
        else TileA[ty][tx] = 0;
        if(Brow < N) {
            for(int j = 0 ; j < multiB ; ++ j) {
                if (Bcol + TileSize * j < K) TileB[ty][tx + TileSize * j] = *(B + Bptr + TileSize * j);
                else TileB[ty][tx + TileSize * j] = 0;
            }
        }

        __syncthreads();
        for(int j = 0 ; j < TileSize ; ++ j) {
            for(int k = 0 ; k < multiB ; ++ k) {
                sum[k] += TileA[ty][j] * TileB[j][tx + TileSize * k];
            }
        }
        __syncthreads();
    }
    int Crow = row + ty, Ccol = col + tx;
    int Cptr = Crow * K + Ccol;
    if(Crow < M) {
        for(int i = 0 ; i < multiB ; ++ i) {
            if(Ccol + TileSize * i < K) {
                C[Cptr + TileSize * i] = sum[i];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TileSize, TileSize);
    dim3 blocksPerGrid((K + (TileSize * multiB) - 1) / (TileSize * multiB),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
