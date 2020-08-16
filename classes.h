#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#include <iostream>
#include <fstream>
#include <iomanip> 

#ifndef KA_CLASSES
#define KA_CLASSES

class Data_sizes {
	size_t uv_kernel_size_in_bytes;
	size_t w_kernel_size_in_bytes;
	size_t subgrid_size_in_bytes;
	size_t visibilities_position_size_in_bytes;
	size_t output_size_in_bytes;
	size_t total_memory_required_in_bytes;
};

class Benchmark_parameters {
	int nRuns;
	int device;
};

class Problem_parameters {
	int uv_kernel_size;
	int uv_kernel_stride;
	int uv_kernel_oversampling;
	int w_kernel_size;
	int w_kernel_stride;
	int w_kernel_oversampling;
	int grid_z;
	int grid_y;
	int grid_x;
	int nVisibilities;
	int nSubgrids;
};

#endif