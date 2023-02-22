## Debugging with CUDA GDB

### Command Line
- Setting `-g` flag to generate debug information for host code, and `-G` flag for device code.  
    ```nvcc -g -G -o out bugfile.cu```
- Start cuda-gdb. Set `CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1` to allow cuda-gdb to break in device.
    ```CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 cuda-gdb ./out```
- Set a breakpoint and set through it. or delete breakpoints.   
    `b add_kernel`   
    `n`   
    `d`
- Print variables or display for every single step.   
    `p <variable name>`   
    `display <variable name>`
- In a kernel, print current thread or block infomation. or change cuda thread or block.  
    `cuda thread`   
    `cuda block`      
    `cuda thread 1`
- Using cuda memory checker.   
    `set cuda memcheck on`

### VS CODE
- [Document](https://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html)
