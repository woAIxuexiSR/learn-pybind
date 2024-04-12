import sys
import torch

def test_my_extension():
    
    sys.path.append("src/build/Debug/")
    import my_extension

    print("Test my_extension")
    print("attr: ", my_extension.hello_world, my_extension.answer)
    print("func: ", my_extension.add(1, 2), my_extension.add(j=1))
    print("class: ", my_extension.A(16).get())

def test_torch_extension():
    
    import torch_extension
    
    print("Test torch_extension")
    a = torch.tensor([1.0, 2.0, 3.0]).cuda()
    b = torch.tensor([4.0, 5.0, 6.0]).cuda()
    print("add: ", torch_extension.tensor_add(a, b))

def test_torch_jit_extension():
    
    from torch.utils.cpp_extension import load
    extension = load(
        name='torch_jit_extension',
        sources=['src/torch_extension.cu'],
        extra_cflags=['-g'],
        extra_cuda_cflags=['-O2'],
        verbose=False
    )
    
    print("Test torch_jit_extension")
    a = torch.tensor([1.0, 2.0, 3.0]).cuda()
    b = torch.tensor([4.0, 5.0, 6.0]).cuda()
    print("add: ", extension.tensor_add(a, b))
    

if __name__ == "__main__":
    
    test_my_extension()
    test_torch_extension()
    test_torch_jit_extension()