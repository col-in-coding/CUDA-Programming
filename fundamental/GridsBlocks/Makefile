CUDAFLAGS ?= -g

# Define a variable
ALL = launch-bounds-1d launch-bounds-2d launch-bounds-2d occupancy-api

# target: prerequisites
all: $(ALL)

# 匹配符：%
# 自动变量：$@ 指代当前目标，$^ 指代所有前置条件prerequisites
%: %.cu
	nvcc -std=c++11 $(CUDAFLAGS) -o $@ $^

