## Debug via Numpy

### Description

Before developing a Tensorrt Plugin, it is recommand to validate the kernel functionality seperatly. Here we generated some numpy data as inputs, then saved the kernel result and compared with numpy calculated result.

cnpy: Numpy data read and write lib from https://github.com/rogersce/cnpy

### Procedure
1. `make clean && make`: compile main files  
2. `python validate.py --gen-inputs`: generate numpy input data  
3. `./main`: get cuda kernel result 
4. `python validate.py -v`: validate the result