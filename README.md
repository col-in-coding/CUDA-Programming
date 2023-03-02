## CUDA 工具

编译器：nvcc(C/C++)

调试器：nvcc-gdb

性能分析：nsight, nvprof

函数库：cublas, nvblas, cusolver, cufftw, cusparse, nvgraph

## Programming with CUDA

### Software - driver and runtime APIs
CUDA Driver API: low-level API，installed via GPU Driver。(libcuda.so)    
Runtime API：High-level API，installed CUDA via toolkit。(libcudart.so, nvcc)

### Exection model - kernels, threads, and blocks
* Kernel - top-level device function 
* Executed by N CUDA threads in parallel 
* Grid, Block, Threads. Threads within the same block share certain resources, and can communicate or synchronize with each other

### Hardware architecture
* Serveral SMs on each GPU; Multiple CUDA cores per SM; Shared cache, registers and memory between the cores; Global memory shared by all SMs
* SIMT Architecture (Single instruction and multiple thread); unlike SIMD, the width here is variable; threads have independent states and can diverge
* Warps (Groups of 32 threads), Occupancy = Active Warps/Maximum allowed warps

### Running a Kernel
* Blocks are assigned to available SMs
* Blocks are split into warps
* Multiple warps/blocks run on each SM
* As blocks finish, new blocks are scheduled

### CUDA API Errors
* cudaError_t   
* Helper functions - cudaGetErrorString(), cudaGetLastError()   


## Performance Optimization

### Memory Hierachy
* Global memory: Large and high latency.   
* L2 cache: Medium latency
* SM caches: Lower latency
* Registers: Lowest latency

### Coalescing memory access
data should be properly aligned and packed together on a 32, 64 or 128 byte boudary, so thread would access adjacent memory location. nvcc would load entire struct (couldn't be large) in one go.
```
struct __align__(16) xxx {
    float a,
    float b,
    float c,
    float d
}
```

2D or 3D Memory Layouts  
使用 `cudaMallocPitch` 保证每行第一个元素的地址是对齐的，这需要补一些其它数值进去，pitch就是加上额外元素之后的实际宽度。  
` T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;`
```
cudaMallocPitch(&input2d.red, &pitch, byte_width, params.height)

cudaMemcpy2D(
    input2d.red, pitch, params.input_image.red,
    byte_width, byte_width, params.height,
    cudaMemcpyDeviceToDevice)

cudaMalloc3D(cudaPitchedPtr* ptr, cudaExtent extent)

cudaMemcpy3D(constcudaMemcpy3DParms* p)
```

### Texture and Constant memory


## Parallel Algorithms

Shared Memory
* 64-96 KB on each SM
* Read/Write access from kernels
* Shared within a thread block

