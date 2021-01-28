cmake_minimum_required(VERSION 3.18)

add_executable(test_bounds test_bounds.cu)
set_property(TARGET test_bounds PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_Bounds test_bounds)

add_executable(test_morton_codes test_morton_codes.cu)
set_property(TARGET test_morton_codes PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_Morton_Codes test_morton_codes)

add_executable(test_radtree_builder test_radtree_builder.cu)
set_property(TARGET test_radtree_builder PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_radtree_builder test_radtree_builder)

add_executable(test_delta test_delta.cu)
set_property(TARGET test_delta PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_delta test_delta)

add_executable(test_determine_direction test_determine_direction.cu)
set_property(TARGET test_determine_direction PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_determine_direction test_determine_direction)

add_executable(test_max_len test_max_len.cu)
set_property(TARGET test_max_len PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_max_len test_max_len)

add_executable(test_len test_len.cu)
set_property(TARGET test_len PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_len test_len)

add_executable(test_find_split test_find_split.cu)
set_property(TARGET test_find_split PROPERTY CUDA_ARCHITECTURES 53)
#add_test(Test_find_split test_find_split)
