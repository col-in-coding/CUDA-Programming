cmake_minimum_required (VERSION 3.9)

PROJECT(PoissonBlend LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Tesla V100: ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70;)

# ADD_EXECUTABLE(PoissonBlend main.cu, lab3.cu, pgm.cpp)
CUDA_ADD_EXECUTABLE( ${PROJECT_NAME}
    main.cu
    lab3.cu
    pgm.cpp
)