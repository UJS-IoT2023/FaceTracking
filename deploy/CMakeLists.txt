cmake_minimum_required(VERSION 3.20)
project(deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

enable_language(CUDA)

set(CMAKE_PREFIX_PATH "/home/cacc/lib/libtorch")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(deploy main.cpp)

target_link_libraries(deploy PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})