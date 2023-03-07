#ifndef LAYERNORM_KERNEL_H
#define LAYERNORM_KERNEL_H

#include <cuda_runtime_api.h>

void computeLayerNorm(
    int32_t const gridSize, int32_t const blockSize,
    float const* input, float const* gamma, float const* beta,
    float* output, const float epsilon);

#endif