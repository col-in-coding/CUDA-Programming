
## Driver API
- Allow more fine grain control of what is executed on device
- Language independent so any basae that can invoke cubin objects will be able to use this code
- The Driver API needs to be initialized at least once via the cuInit function
- Will need to manage devices, modules, and contexts, especially if code needs to interact with higher level

Compile to ptx file

```
nvcc -o matSumKernel.ptx -ptx matSumKernel.cu -lcuda
```

Compile to fatbin file

```
nvcc -o matSumKernel.fatbin -fatbin matSumKernel.cu -lcuda
```

### Compile CPP file

```
nvcc -o drivertest.o -c drivertest.cpp -lcuda
nvcc -o drivertest drivertest.o -lcuda
```

## Runtime API
- Removes the need for keeping track of contexts, certain forms of initialization and management of modules
- No need to load specific kernels as all available kernels are initialized on program start
- Specific for C++

Compile CU file

```
nvcc -o vector_add vector_add.cu
```