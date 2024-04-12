cmake_minimum_required(VERSION 3.20)

project(pybind LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_subdirectory(../pybind11 ${CMAKE_BINARY_DIR}/pybind11)
pybind11_add_module(my_extension my_extension.cu)