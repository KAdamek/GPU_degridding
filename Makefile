INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcuda -lcufft
# -lfftw3f -lgsl -lgslcblas

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = SKA_degrid.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fastmath
NVCCFLAGS += --use_fast_math
endif

all: clean analyze

analyze: SKA_degrid.o GPU_SKA_degrid.o Makefile
	$(NVCC) -o $(ANALYZE) SKA_degrid.o GPU_SKA_degrid.o  $(LIB) $(NVCCFLAGS) 

GPU_SKA_degrid.o: timer.h utils_cuda.h
	$(NVCC) -c -dc GPU_SKA_degrid.cu $(NVCCFLAGS)
	
SKA_degrid.o: SKA_degrid.cpp
	$(GCC) -c SKA_degrid.cpp $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


