#include <cuda_runtime.h>
#include "helper_string.h"
#include "helper_cuda.h"

#include <thrust/device_vector.h>

#include <iostream>

__global__ void transpose_kernel(int *a, int *out, int row, int col, int tileSize)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int rIdx = blockIdx.x * tileSize;
    int cIdx = blockIdx.y * tileSize;

    for (int i = 0; i < tileSize; i++)
    {
        int r = rIdx + tid, c = cIdx + i;
        if(r < row && c < col)
            sdata[tid * (tileSize + 1) + i] = a[r * col + c];
    }
    __syncthreads();

    for (int i = 0; i < tileSize; i++)
    {
        int r = cIdx + tid, c = rIdx + i;
        if (r < col && c < row)
            out[r * row + c] = sdata[i * (tileSize + 1) + tid];
    }

    // direct transpose, why is it faster?
    // for(int i = 0; i < tileSize; i++)
    // {
    //     int idx = (rIdx + tid) * col + cIdx + i;
    //     int oidx = (cIdx + i) * row + rIdx + tid;
    //     if(rIdx + tid < row && cIdx + i < col)
    //         out[oidx] = a[idx];
    // }
}

// divide into tileSize x tileSize blocks, use tileSize x (tileSize + 1) to avoid bank conflicts
void transpose(int *a, int *out, int row, int col, int tileSize = 32)
{
    int rBlocks = (row + tileSize - 1) / tileSize;
    int cBlocks = (col + tileSize - 1) / tileSize;
    int shared_memory_size = tileSize * (tileSize + 1) * sizeof(int);
    dim3 grid(rBlocks, cBlocks);
    transpose_kernel<<<grid, tileSize, shared_memory_size>>>(a, out, row, col, tileSize);
    checkCudaErrors(cudaDeviceSynchronize());
}

int main()
{
    const int row = 10000, col = 25000;
    const int N = row * col;
    thrust::host_vector<int> h_v(N, 0);
    for (int i = 0; i < N; i++)
        h_v[i] = i;
    thrust::device_vector<int> d_v = h_v;
    thrust::device_vector<int> out(N, 0);

    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    checkCudaErrors(cudaEventRecord(start));

    {
        transpose(thrust::raw_pointer_cast(d_v.data()),
                  thrust::raw_pointer_cast(out.data()),
                  row, col);
    }

    checkCudaErrors(cudaEventRecord(end));

    checkCudaErrors(cudaEventSynchronize(end));
    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    printf("Time: %f ms\n", ms);

    h_v = out;
    bool correct = true;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            if (h_v[i * row + j] != j * col + i)
                correct = false;
        }
    printf("Correct: %s\n", correct ? "true" : "false");

    return 0;
}