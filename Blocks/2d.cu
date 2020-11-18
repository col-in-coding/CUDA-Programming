// Demonstration of kernel execution configuration for a two-dimensional
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

__global__ void kernel_2d()
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("block (%d, %d), thread (%d, %d), index (%d, %d)\n", blockIdx.x,
         blockIdx.y, threadIdx.x, threadIdx.y, x, y);
}

int main()
{
  dim3 block_dim(8, 2);
  dim3 grid_dim(2, 1);
  kernel_2d<<<grid_dim, block_dim>>>();
  cudaCheckError(cudaDeviceSynchronize());
}

// Output:
// block (0, 0), thread (0, 0), index (0, 0)
// block (0, 0), thread (1, 0), index (1, 0)
// block (0, 0), thread (2, 0), index (2, 0)
// block (0, 0), thread (3, 0), index (3, 0)
// block (0, 0), thread (4, 0), index (4, 0)
// block (0, 0), thread (5, 0), index (5, 0)
// block (0, 0), thread (6, 0), index (6, 0)
// block (0, 0), thread (7, 0), index (7, 0)
// block (0, 0), thread (0, 1), index (0, 1)
// block (0, 0), thread (1, 1), index (1, 1)
// block (0, 0), thread (2, 1), index (2, 1)
// block (0, 0), thread (3, 1), index (3, 1)
// block (0, 0), thread (4, 1), index (4, 1)
// block (0, 0), thread (5, 1), index (5, 1)
// block (0, 0), thread (6, 1), index (6, 1)
// block (0, 0), thread (7, 1), index (7, 1)
// block (1, 0), thread (0, 0), index (8, 0)
// block (1, 0), thread (1, 0), index (9, 0)
// block (1, 0), thread (2, 0), index (10, 0)
// block (1, 0), thread (3, 0), index (11, 0)
// block (1, 0), thread (4, 0), index (12, 0)
// block (1, 0), thread (5, 0), index (13, 0)
// block (1, 0), thread (6, 0), index (14, 0)
// block (1, 0), thread (7, 0), index (15, 0)
// block (1, 0), thread (0, 1), index (8, 1)
// block (1, 0), thread (1, 1), index (9, 1)
// block (1, 0), thread (2, 1), index (10, 1)
// block (1, 0), thread (3, 1), index (11, 1)
// block (1, 0), thread (4, 1), index (12, 1)
// block (1, 0), thread (5, 1), index (13, 1)
// block (1, 0), thread (6, 1), index (14, 1)
// block (1, 0), thread (7, 1), index (15, 1)