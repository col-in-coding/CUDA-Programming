### Compile to ptx file

```
nvcc -o matSumKernel.ptx -ptx matSumKernel.cu -lcuda
```

### Compile to fatbin file

```
nvcc -o matSumKernel.fatbin -fatbin matSumKernel.cu -lcuda
```

### Compile CPP file

```
nvcc -o drivertest.o -c drivertest.cpp -lcuda
nvcc -o drivertest drivertest.o -lcuda
```