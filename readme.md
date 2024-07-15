Call C++ from Python
====================

1. Simple pybind11

   `cmake` build src file `my_extension.cu` and add output to path

2. Torch extension

   `python setup.py install` build src file `troch_extension.cu`

3. Torch JIT

   compile `torch_extension.cu` just in time


Extra CUDA optimization in `src`:

- reduce
- matrix transpose
- matrix multiplication