CUDAFLAGS ?= -g -G
LIBS = `pkg-config --libs zlib`

ALL = main

all: $(ALL)

%: %.cpp cnpy.o layernormKernel.o
	nvcc -std=c++11 $(CUDAFLAGS) -o $@ $^ $(LIBS)

cnpy.o: cnpy.cpp cnpy.h
	g++ -std=c++11 $(CUDAFLAGS) -o $@ -c $<

layernormKernel.o: layernormKernel.cu layernormKernel.h
	nvcc -std=c++11 $(CUDAFLAGS) -o $@ -c $<

.PHONY: clean
clean:
	rm -f *.o $(ALL)