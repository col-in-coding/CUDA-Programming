#include <iostream>

#include "cnpy.h"
#include "layernormKernel.h"

int main()
{
    cnpy::NpyArray inpArr = cnpy::npy_load("./inp.npy");
    cnpy::NpyArray gammaArr = cnpy::npy_load("./gamma.npy");
    cnpy::NpyArray betaArr = cnpy::npy_load("./beta.npy");
    std::vector<size_t> out_shape(inpArr.shape);
    cnpy::NpyArray outArr(out_shape, inpArr.word_size, inpArr.fortran_order);
    /*
    std::cout << "word size: " << inpArr.word_size << std::endl;
    std::cout << "shape: (";
    for (int i = 0; i < inpArr.shape.size(); i++)
    {
        std::cout << inpArr.shape[i] << ", ";
    }
    std::cout << ")" << std::endl;
    std::cout << "num vals: " << inpArr.num_vals << std::endl;
    */

    float *pInput, *gamma, *beta, *pOutput;
    int inp_size = inpArr.num_vals * inpArr.word_size;
    size_t gamma_size = gammaArr.num_vals * gammaArr.word_size;
    size_t beta_size = betaArr.num_vals * betaArr.word_size;
    cudaMalloc((void **)&pInput, inp_size);
    cudaMalloc((void **)&gamma, gamma_size);
    cudaMalloc((void **)&beta, beta_size);
    cudaMalloc((void **)&pOutput, inp_size);
    cudaMemcpy(pInput, inpArr.data<float>(), inp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gamma, gammaArr.data<float>(), gamma_size, cudaMemcpyHostToDevice);
    cudaMemcpy(beta, betaArr.data<float>(), beta_size, cudaMemcpyHostToDevice);

    const int32_t nBlock = inpArr.shape[0] * inpArr.shape[1];
    const int32_t nThread = inpArr.shape[2] / 3;

    computeLayerNorm(nBlock, nThread, pInput, gamma, beta, pOutput, 6e-6);

    cudaMemcpy(outArr.data<float>(), pOutput, inp_size, cudaMemcpyDeviceToHost);

    cnpy::npy_save("out.npy", outArr.data<float>(), outArr.shape); //"w" overwrites any existing file
  
    cudaFree(pInput);
    cudaFree(gamma);
    cudaFree(beta);
    cudaFree(pOutput);
    return 0;
}