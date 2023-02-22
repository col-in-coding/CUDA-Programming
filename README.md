## CUDA 工具

编译器：nvcc(C/C++)

调试器：nvcc-gdb

性能分析：nsight, nvprof

函数库：cublas, nvblas, cusolver, cufftw, cusparse, nvgraph

## Programming

### Software - driver and runtime APIs
CUDA Driver API: low-level API 太底层使用不方便，随GPU Driver安装。(libcuda.so)    
Runtime API：High-level API，随CUDA toolkit安装。(libcudart.so, nvcc)

### Exection model - kernels, threads, and blocks
* Kernel - top-level device function 
* Executed by N CUDA threads in parallel 
* Grid, Block, Threads. Threads within the same block share certain resources, and can communicate or synchronize with each other

### Hardware architecture
* Serveral SMs on each GPU; Multiple CUDA cores per SM; Shared cache, registers and memory between the cores; Global memory shared by all SMs
* SIMT Architecture (Single instruction and multiple thread); unlike SIMD, the width here is variable; threads have independent states and can diverge
* Warps (Groups of 32 threads), Occupancy = Active Warps/Maximum allowed warps

### Running a Kernel
1. Blocks are assigned to available SMs
2. Blocks are split into warps
3. Multiple warps/blocks run on each SM
4. As blocks finish, new blocks are scheduled

## Performance Optimization

### Memory Hierachy
* Global memory: Large and high latency.<br /> 
    <i>Thead-level parallelism<br /></i>
    <i>Coalescing memory access: all data for the warp load in register, as long as the threads need it. ( \_\_align__() , )</i>
* L2 cache: Medium latency
* SM caches: Lower latency
* Registers: Lowest latency

### Alignment
data should be properly aligned and packed together on a 32, 64 or 128 byte boudary, so thread would access adjacent memory location. nvcc would load entire struct (couldn't be large) in one go.
```
struct \_\_align__() xxx {
    float a,
    float b,
    ...
}
```

2D or 3D Memory Layouts
```
cudaMallocPitch(&input2d.red, &pitch, byte_width, params.height)

cudaMemcpy2D(
    input2d.red, pitch, params.input_image.red,
    byte_width, byte_width, params.height,
    cudaMemcpyDeviceToDevice)

cudaMalloc3D(cudaPitchedPtr* ptr, cudaExtent extent)

cudaMemcpy3D(constcudaMemcpy3DParms* p)
```

## Parallel Algorithms
Shared Memory
* 64-96 KB on each SM
* Read/Write access from kernels
* Shared within a thread block

### Reduction

### Prefix Sum

### Filtering# CUDA-Programming
