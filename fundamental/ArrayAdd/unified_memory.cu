#include <assert.h>
#include <stdio.h>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// Error checking macro
#define cudaCheckError(code)                                                   \
    {                                                                          \
        if ((code) != cudaSuccess)                                             \
        {                                                                      \
            fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
                    cudaGetErrorString(code));                                 \
        }                                                                      \
    }

// Device kernel for array addition.
__global__ void add_kernel(float *dest, int n_elts, const float *a,
                           const float *b)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_elts)
        return;
    // printf("(%d, %d) \n", blockIdx.x, threadIdx.x);
    dest[index] = a[index] + b[index];
}

int main()
{
    const int ARRAY_LENGTH = 1 << 20;

    float *a, *b, *c;

    // Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&a, ARRAY_LENGTH*sizeof(float));
    cudaMallocManaged(&b, ARRAY_LENGTH*sizeof(float));
    cudaMallocManaged(&c, ARRAY_LENGTH*sizeof(float));

    for (int i = 0; i < ARRAY_LENGTH; i++)
    {
        a[i] = 2 * i;
        b[i] = 2 * i + 1;
    }

    // Calculate lauch configuration
    const int BLOCK_SIZE = 512;
    int n_blocks = (ARRAY_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Add arrays on device,
    // Kernels are the top-level functions used to run code on the device.
    add_kernel<<<n_blocks, BLOCK_SIZE>>>(c, ARRAY_LENGTH, a, b);

    // for (int i = 0; i < ARRAY_LENGTH; i++)
    // {
    //     printf("%g + %g = %g\n", a[i], b[i], c[i]);
    // }

    // Free Memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
