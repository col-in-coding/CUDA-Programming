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
  // 3.2 版本以上的 SDK 支持在内核中printf
  printf("block %d, thread %d, index %d\n", blockIdx.x, threadIdx.x, index);
}

int main()
{
  // 对应Grid dimension与Block dimension直接输入int，表示维度为1
  kernel_1d<<<4, 8>>>();
  cudaCheckError(cudaDeviceSynchronize());
}
