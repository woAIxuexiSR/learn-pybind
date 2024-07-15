#include <cuda_runtime.h>
#include "helper_string.h"
#include "helper_cuda.h"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <iostream>

__global__ void reduce_kernel(int *a, int *out, int N, int k)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * k + threadIdx.x;

    sdata[tid] = 0;
    for(int i = 0; i < k * blockDim.x; i += blockDim.x)
        sdata[tid] += (idx + i < N) ? a[idx + i] : 0;
    __syncthreads();

    for (int d = blockDim.x / 2; d > 0; d >>= 1)
    {
        if (tid < d)
            sdata[tid] += sdata[tid + d];
        __syncthreads();
    }
    // TODO: unroll loops

    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

// divide into blocks that each reduce (numThreads * k) data
int reduce(int *a, int N, int numThreads = 256, int k = 16)
{
    int numData = numThreads * k;
    int numBlocks = (N + numData - 1) / numData;

    int *out;
    checkCudaErrors(cudaMalloc(&out, numBlocks * sizeof(int)));
    reduce_kernel<<<numBlocks, numThreads, numThreads * sizeof(int)>>>(a, out, N, k);

    N = numBlocks;
    while (N > 1)
    {
        numBlocks = (N + numData - 1) / numData;
        reduce_kernel<<<numBlocks, numThreads, numThreads * sizeof(int)>>>(out, out, N, k);
        N = numBlocks;
    }
    
    int result;
    checkCudaErrors(cudaMemcpy(&result, out, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(out));
    return result;
}

int main()
{
    const int N = 10000000;
    thrust::host_vector<int> h_v(N, 1);
    thrust::device_vector<int> d_v = h_v;

    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    checkCudaErrors(cudaEventRecord(start));

    {

        // int sum = thrust::reduce(d_v.begin(), d_v.end(), 0);
        // printf("Sum: %d\n", sum);

        int sum = reduce(thrust::raw_pointer_cast(d_v.data()), N);
        printf("Sum : %d\n", sum);
    }

    checkCudaErrors(cudaEventRecord(end));

    checkCudaErrors(cudaEventSynchronize(end));
    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    printf("Time: %f ms\n", ms);

    return 0;
}