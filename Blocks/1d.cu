// Demonstration of kernel execution configuration for a one-dimensional
// grid.

#include <cuda_runtime_api.h>
#include <stdio.h>

// Error checking macro
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }

__global__ void kernel_1d()
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("block %d, thread %d, index %d\n", blockIdx.x, threadIdx.x, index);
}

int main()
{
  kernel_1d<<<4, 8>>>();
  cudaCheckError(cudaDeviceSynchronize());
}

// Output:
// block 0, thread 0, index 0
// block 0, thread 1, index 1
// block 0, thread 2, index 2
// block 0, thread 3, index 3
// block 0, thread 4, index 4
// block 0, thread 5, index 5
// block 0, thread 6, index 6
// block 0, thread 7, index 7
// block 1, thread 0, index 8
// block 1, thread 1, index 9
// block 1, thread 2, index 10
// block 1, thread 3, index 11
// block 1, thread 4, index 12
// block 1, thread 5, index 13
// block 1, thread 6, index 14
// block 1, thread 7, index 15
// block 3, thread 0, index 24
// block 3, thread 1, index 25
// block 3, thread 2, index 26
// block 3, thread 3, index 27
// block 3, thread 4, index 28
// block 3, thread 5, index 29
// block 3, thread 6, index 30
// block 3, thread 7, index 31
// block 2, thread 0, index 16
// block 2, thread 1, index 17
// block 2, thread 2, index 18
// block 2, thread 3, index 19
// block 2, thread 4, index 20
// block 2, thread 5, index 21
// block 2, thread 6, index 22
// block 2, thread 7, index 23