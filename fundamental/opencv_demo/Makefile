CUDAFLAGS ?= -g
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

ALL = mat

all: $(ALL)

%: %.cpp
	g++ -std=c++11 $(CFLAGS) $(CUDAFLAGS) -o $@ $< $(LIBS)

