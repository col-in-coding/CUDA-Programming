#include <stdio.h>

__global__ void gpu_shared_memory(float *d_a)
{
    int i, index = threadIdx.x;
    float average, sum = 0.0f;
    // Defining shared_memory
    __shared__ float sh_arr[10];
    sh_arr[index] = d_a[index];
    // ensure all the writes to shared memory have completed
    __syncthreads();
    for (i = 0; i <= index; i++)
    {
        sum += sh_arr[i];
    }
    average = sum / (index + 1.0f);
    d_a[index] = average;
}

int main(int argc, char **argv)
{
    float h_a[10];
    float *d_a;
    // initialize host Array
    for (int i = 0; i < 10; i++)
    {
        h_a[i] = i;
    }
    // allocate global memory on the device
    cudaMalloc((void **)&d_a, sizeof(float) * 10);
    // copy data from host memory to device memory
    cudaMemcpy((void *)d_a, (void *)h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);
    gpu_shared_memory<<<1, 10>>>(d_a);
    // copy the modified array back to the host
    cudaMemcpy((void *)h_a, (void *)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    printf("Use of shared memory on GPU: \n");
    for(int i = 0; i < 10; i++)
    {
        printf("The running average after %d element is %f \n", i, h_a[i]);
    }
}