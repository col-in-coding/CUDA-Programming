## How to compile and run the code

`nvcc -o array-add ./main.cu`

`./array-add`

## Memory Allocation

allocate an independent GPU memory
```C
float *device_a;
cudaMalloc(&device_a, size);
cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
...
cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);
```

allocate a Unified Memory
```C
float *a;
cudaMallocManaged(&a, size); // done!
```

## CUDA Kernel

Symtax
```C
// Define a device kernal
__global__
void add_kernel(float *c, int len, float *a, float *b)
{...}
// Invoke
add_kernel<<<numBlocks, threadsPerBlock>>>(c, len, a, b);
```

## Profile

command
```Bash
nsys profile --stats=true ./array-add
```

results on 3090
```Bash
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)     GridXYZ         BlockXYZ                              Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  --------------  --------------
    100.0          6813004          1  6813004.0  6813004.0   6813004   6813004          0.0  2048    1    1   512    1    1  add_kernel(float *, int, const float *, const float *)
```