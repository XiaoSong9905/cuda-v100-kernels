cmake_minimum_required(VERSION 3.10)
project(reduce-v100)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(main main.cu )
target_compile_features(main PRIVATE cxx_std_11)
target_include_directories(main PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(STATUS "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -gencode=arch=compute_70,code=sm_70 ")

# Copy the job scripts
# configure_file(job-gpu job-gpu COPYONLY)
