#include "layernormKernel.h"

__global__ void layerNormKernel(float const *pInput, float const *gamma, float const *beta, float *pOutput, const float epsilon)
{
    // nDim=768
    const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

    __shared__ float temp[256];

    float value0 = pInput[index];
    float value1 = pInput[index + 256];
    float value2 = pInput[index + 512];

    temp[tx] = value0 + value1 + value2;
    __syncthreads();

    for (int stride = 128; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float mean = temp[0] / 768;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean) + (value2 - mean) * (value2 - mean);
    __syncthreads();

    for (int stride = 128; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / 768;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6) * gamma[tx] + beta[tx];
    pOutput[index + 256] = (value1 - mean) * rsqrtf(var + 6e-6) * gamma[tx + 256] + beta[tx + 256];
    pOutput[index + 512] = (value2 - mean) * rsqrtf(var + 6e-6) * gamma[tx + 512] + beta[tx + 512];
}

void computeLayerNorm(
    int32_t const gridSize, int32_t const blockSize,
    float const* input, float const* gamma, float const* beta,
    float* output, const float epsilon)
{
    layerNormKernel<<<gridSize, blockSize>>> (
        input, gamma, beta, output, epsilon
    );
}