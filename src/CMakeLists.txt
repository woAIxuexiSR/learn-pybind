cmake_minimum_required(VERSION 3.20)

project(pybind LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_subdirectory(../pybind11 ${CMAKE_BINARY_DIR}/pybind11)
add_subdirectory(../tiny-cuda-nn ${CMAKE_BINARY_DIR}/tiny-cuda-nn)

pybind11_add_module(my_extension my_extension.cu)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(reduce reduce.cu)
add_executable(matrix_transpose matrix_transpose.cu)
add_executable(matmul matmul.cu)