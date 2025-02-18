cmake_minimum_required(VERSION 3.16)
project(samoyeds_kernel)

set(CUDA_TOOLKIT_PATH "/usr/local/cuda/")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_PATH}/bin/nvcc")
# set(CMAKE_BUILD TYPE Release)

find_package (Python COMPONENTS Interpreter Development)
set(pybind11_DIR "${Python_SITELIB}/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# set(CMAKE_CUDA_FLAGS "-G")
# set(CUDA_NVCC_FLAGS -G;-g)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(SAMOYEDS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Samoyeds-Kernel/benchmark")

add_subdirectory(Samoyeds-Kernel)

# This line may clear the cache for CMAKE_CUDA_ARCHITECTURES
# Make sure this line is executed after add_subdirectory(Samoyeds-Kernel)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(samoyeds_mod)
