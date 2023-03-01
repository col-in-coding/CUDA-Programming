CUDAFLAGS ?= -g
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

ALL = img_test

all: $(ALL)

%: %.cu
	nvcc -std=c++11 $(CFLAGS) $(CUDAFLAGS) -o $@ $< $(LIBS)

