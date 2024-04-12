import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_extension",
    ext_modules=[
        CUDAExtension(
            name="torch_extension",
            sources=["src/torch_extension.cu"],
            include_dirs=torch.utils.cpp_extension.include_paths(),
            library_dirs=torch.utils.cpp_extension.library_paths(),
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# run `python setup.py install` to install the extension
# use `pip uninstall torch_extension` to uninstall the extension
