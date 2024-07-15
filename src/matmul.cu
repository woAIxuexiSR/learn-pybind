#include <cuda_runtime.h>
#include "helper_string.h"
#include "helper_cuda.h"

#include <thrust/device_vector.h>

#include <iostream>

__global__ void matmul_kernel(int *a, int *b, int *out, int M, int N, int K, int tileSize)
{
    extern __shared__ int sdata[];

    int tileStride = tileSize + 1;
    int *As = sdata, *Bs = sdata + tileSize * tileStride;

    int tid = threadIdx.x;
    int rIdx = blockIdx.x * tileSize;
    int cIdx = blockIdx.y * tileSize;
    
    for(int i = 0; i < tileSize; i++)
    {
        int idx = (rIdx + tid) * N + cIdx + i;
        bool valid = (rIdx + tid < M) && (cIdx + i < N);
        if(valid) out[idx] = 0;
    }

    int kStep = (K + tileSize - 1) / tileSize;
    for (int k = 0; k < kStep; k++)
    {
        int kIdx = k * tileSize;
        for (int i = 0; i < tileSize; i++)
        {
            int idx = (rIdx + tid) * K + kIdx + i;
            bool valid = (rIdx + tid < M) && (kIdx + i < K);
            As[tid * tileStride + i] = valid ? a[idx] : 0;
        }
        for(int i = 0; i < tileSize; i++)
        {
            int idx = (kIdx + tid) * N + cIdx + i;
            bool valid = (kIdx + tid < K) && (cIdx + i < N);
            Bs[tid * tileStride + i] = valid ? b[idx] : 0;
        }
        __syncthreads();

        for(int i = 0; i < tileSize; i++)
        {
            int idx = (rIdx + tid) * N + cIdx + i;
            bool valid = (rIdx + tid < M) && (cIdx + i < N);
            if(!valid) continue;
            for(int j = 0; j < tileSize; j++)
                out[idx] += As[tid * tileStride + j] * Bs[j * tileStride + i];
        }
        __syncthreads();
    }

    // direct matmul
    // for (int i = rIdx; i < rIdx + tileSize; i++)
    // {
    //     for (int j = cIdx; j < cIdx + tileSize; j++)
    //     {
    //         if (i < M && j < N)
    //         {
    //             int sum = 0;
    //             for (int k = 0; k < K; k++)
    //                 sum += a[i * K + k] * b[k * N + j];
    //             out[i * N + j] = sum;
    //         }
    //     }
    // }
}

// a: M x K, b: K x N, out: M x N
void matmul(int *a, int *b, int *out, int M, int N, int K, int tileSize = 32)
{
    int rBlocks = (M + tileSize - 1) / tileSize;
    int cBlocks = (N + tileSize - 1) / tileSize;
    int shared_memory_size = tileSize * (tileSize + 1) * 2 * sizeof(int);
    dim3 grid(rBlocks, cBlocks);
    matmul_kernel<<<grid, tileSize, shared_memory_size>>>(a, b, out, M, N, K, tileSize);
    checkCudaErrors(cudaDeviceSynchronize());
}

int main()
{
    const int M = 2000, K = 1500, N = 1679;
    thrust::host_vector<int> h_a(M * K, 0), h_b(K * N, 0);
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() % 100;
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() % 100;
    thrust::host_vector<int> h_o(M * N, 0);
    thrust::device_vector<int> d_a = h_a, d_b = h_b, d_o = h_o;

    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    checkCudaErrors(cudaEventRecord(start));

    {
        matmul(thrust::raw_pointer_cast(d_a.data()),
               thrust::raw_pointer_cast(d_b.data()),
               thrust::raw_pointer_cast(d_o.data()),
               M, N, K);
    }

    checkCudaErrors(cudaEventRecord(end));

    checkCudaErrors(cudaEventSynchronize(end));
    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    printf("Time: %f ms\n", ms);

    h_o = d_o;
    bool correct = true;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < K; k++)
                sum += h_a[i * K + k] * h_b[k * N + j];
            if (h_o[i * N + j] != sum)
                correct = false;
        }
    }
    printf("Correct: %s\n", correct ? "true" : "false");

    return 0;
}