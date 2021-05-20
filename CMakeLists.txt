cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(radtree CUDA)

set(CMAKE_CUDA_STANDARD 17)

#enable_testing()
#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
#endif()

#add_compile_options(-O3)
# add_executable(radtree main.cu)

add_library(radtree INTERFACE)
target_include_directories(radtree INTERFACE include/)

# 53 for jetson nano
set_property(TARGET radtree PROPERTY CUDA_ARCHITECTURES 53)

set_target_properties(
        radtree
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)