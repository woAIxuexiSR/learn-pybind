#include <torch/extension.h>

__global__ void cuda_kernel(float *a, float* b, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    output[idx] = a[idx] + b[idx];
}

torch::Tensor tensor_add(torch::Tensor a, torch::Tensor b) {
    auto output = torch::empty_like(a);
    cuda_kernel<<<(a.numel() + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), a.numel());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_add", &tensor_add, "tensor add");
}