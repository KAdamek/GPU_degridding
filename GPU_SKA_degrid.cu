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

using namespace std;


class SKA_degrid_params {
public:
	static const int nRows_per_thread = 1;
	static const int warp = 32;
};

#define NTHREADS 256
#define Y_STEPS 4
#define X_STEPS 4
#define HALF_WARP 16
#define NSEEDS 32
#define WARP 32
#define BUFFER 32

// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************

__inline__ __device__ void calculate_coordinates(
		int grid_size, //dimension of the image's subgrid grid_size x grid_size x 4?
		int kernel_size, // gcf kernel support
		int kernel_stride, // padding of the gcf kernel
		int oversample, // oversampling of the uv kernel
		int wkernel_size, // gcf in w kernel support
		int wkernel_stride, // padding of the gcf w kernel
		int oversample_w, // oversampling of the w kernel
		double theta, //conversion parameter from uv coordinates to xy coordinates x=u*theta
		double wstep, //conversion parameter from w coordinates to z coordinates z=w*wstep 
		double u, // 
		double v, // coordinates of the visibility 
		double w, //
		int *grid_offset, // offset in the image subgrid
		int *sub_offset_x, //
		int *sub_offset_y, // fractional coordinates
		int *sub_offset_z //
	){
	// x coordinate
	double x = theta*u;
	double ox = x*oversample;
	//int iox = lrint(ox);
	int iox = round(ox); // round to nearest
    iox += (grid_size / 2 + 1) * oversample - 1;
    int home_x = iox / oversample;
    int frac_x = oversample - 1 - (iox % oversample);
	
	// y coordinate
	double y = theta*v;
	double oy = y*oversample;
	//int iox = lrint(ox);
	int ioy = round(oy);
    ioy += (grid_size / 2 + 1) * oversample - 1;
    int home_y = ioy / oversample;
    int frac_y = oversample - 1 - (ioy % oversample);
	
	// w coordinate
	double z = 1.0 + w/wstep;
	double oz = z*oversample_w;
	//int iox = lrint(ox);
	int ioz = round(oz);
    ioz += oversample_w - 1;
    //int home_z = ioz / oversample_w;
    int frac_z = oversample_w - 1 - (ioz % oversample_w);
	
    *grid_offset = (home_y-kernel_size/2)*grid_size + (home_x-kernel_size/2);
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
}

__device__ __inline__ double2 Reduce_SM(double2 *s_data){
	double2 l_A = s_data[threadIdx.x];
	
	for (int i = ( blockDim.x >> 1 ); i > HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			l_A.x = s_data[threadIdx.x].x + s_data[i + threadIdx.x].x;
			l_A.y = s_data[threadIdx.x].y + s_data[i + threadIdx.x].y;
			s_data[threadIdx.x] = l_A;
		}
		__syncthreads();
	}
	
	return(l_A);
}

__device__ __inline__ void Reduce_WARP(double2 *A){
	double2 l_A;
	
	for (int q = HALF_WARP; q > 0; q = q >> 1) {
		l_A.x = __shfl_down_sync(0xFFFFFFFF, (*A).x, q);
		l_A.y = __shfl_down_sync(0xFFFFFFFF, (*A).y, q);
		__syncwarp();
		(*A).x = (*A).x + l_A.x;
		(*A).y = (*A).y + l_A.y;
	}
}



template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk1(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		int uv_kernel_size, 
		int uv_kernel_stride, 
		int uv_kernel_oversampling, 
		double *d_gcf_w_kernel, 
		int w_kernel_size, 
		int w_kernel_stride, 
		int w_kernel_oversampling, 
		double2 *d_subgrid, 
		int grid_z, 
		int grid_y, 
		int grid_x, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int nVisibilities, 
		double theta,
		double wstep
	){
	extern __shared__ double2 s_local[];
	
	int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
	calculate_coordinates(
		grid_x,
		uv_kernel_size, uv_kernel_stride, uv_kernel_oversampling,
		w_kernel_stride, w_kernel_stride, w_kernel_oversampling,
		theta, wstep, 
		d_u_vis_pos[blockIdx.y*nVisibilities + blockIdx.x], 
		d_v_vis_pos[blockIdx.y*nVisibilities + blockIdx.x], 
		d_w_vis_pos[blockIdx.y*nVisibilities + blockIdx.x],
		&grid_offset, 
		&sub_offset_x, &sub_offset_y, &sub_offset_z
	);
	
	double2 vis;
	vis.x = 0; vis.y = 0;
	int local_x = (threadIdx.x&7);
	int local_y = (threadIdx.x>>3);
	
	for (int z = 0; z < w_kernel_size; z++) {
		double2 grid_value = d_subgrid[blockIdx.y*grid_z*grid_y*grid_x + z*grid_x*grid_y + grid_offset + local_y*grid_y + local_x];
		
		vis.x += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.x;
		vis.y += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.y;
	}
	
	// NOTE: Reduction checked
	s_local[threadIdx.x] = vis;
	__syncthreads();
	double2 sum;
	sum = Reduce_SM(s_local);
	Reduce_WARP(&sum);	
	__syncthreads();
	
	if(threadIdx.x==0) d_output_visibilities[blockIdx.y*nVisibilities + blockIdx.x] = sum;
}



void SKA_init(){
	//---------> Specific nVidia stuff
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
}


int SKA_degrid_benchmark(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		int uv_kernel_size, 
		int uv_kernel_stride, 
		int uv_kernel_oversampling, 
		double *d_gcf_w_kernel, 
		int w_kernel_size, 
		int w_kernel_stride, 
		int w_kernel_oversampling, 
		double2 *d_subgrid, 
		int grid_z, 
		int grid_y, 
		int grid_x, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int nVisibilities, 
		int nSubgrids,
		double theta,
		double wstep,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize(nVisibilities, nSubgrids, 1);
	dim3 blockSize(uv_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = uv_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Pulse detection FIR
	SKA_init();
	GPU_SKA_degrid_kernel_mk1<SKA_degrid_params><<< gridSize , blockSize, shared_mem >>>(
		d_output_visibilities, 
		d_gcf_uv_kernel, 
		uv_kernel_size, 
		uv_kernel_stride, 
		uv_kernel_oversampling, 
		d_gcf_w_kernel, 
		w_kernel_size, 
		w_kernel_stride, 
		w_kernel_oversampling, 
		d_subgrid, 
		grid_z, 
		grid_y, 
		grid_x, 
		d_u_vis_pos, 
		d_v_vis_pos, 
		d_w_vis_pos, 
		nVisibilities, 
		theta,
		wstep
	);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}


int check_memory(size_t total_size, float multiple){
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	double free_memory     = ((double) free_mem);
	double required_memory = multiple*((double) total_size);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", ((float) total_mem)/(1024.0*1024.0), free_memory/(1024.0*1024.0), required_memory/(1024.0*1024.0));
	if(required_memory>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(3);}
	return(0);
}


int GPU_SKA_degrid(
		double2 *h_output_visibilities, 
		double *h_gcf_uv_kernel, 
		int uv_kernel_size, 
		int uv_kernel_stride, 
		int uv_kernel_oversampling, 
		double *h_gcf_w_kernel, 
		int w_kernel_size, 
		int w_kernel_stride, 
		int w_kernel_oversampling, 
		double2 *h_subgrid, 
		int grid_z, 
		int grid_y, 
		int grid_x, 
		double *h_u_vis_pos, 
		double *h_v_vis_pos, 
		double *h_w_vis_pos, 
		int nVisibilities, 
		int nSubgrids,
		double theta,
		double wstep,
		int nRuns,
		int device,
		double *execution_time
	) {
	//---------> Initial nVidia stuff
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount) checkCudaErrors(cudaSetDevice(device));
	else { printf("Wrong device!\n"); exit(1); }
	
	size_t uv_kernel_size_in_bytes             = uv_kernel_stride*uv_kernel_oversampling*sizeof(double);
	size_t w_kernel_size_in_bytes              = w_kernel_stride*w_kernel_oversampling*sizeof(double);
	size_t subgrid_size_in_bytes               = grid_z*grid_y*grid_x*sizeof(double2)*nSubgrids;
	size_t visibilities_position_size_in_bytes = nVisibilities*sizeof(double)*nSubgrids;
	size_t output_size_in_bytes                = nVisibilities*sizeof(double2)*nSubgrids;
	size_t total_memory_required_in_bytes      = uv_kernel_size_in_bytes + w_kernel_size_in_bytes + subgrid_size_in_bytes + visibilities_position_size_in_bytes + output_size_in_bytes;
	
	//---------> Checking memory
	if(check_memory(total_memory_required_in_bytes, 1.0)!=0) return(1);
	
	//---------> Measurements
	double exec_time = 0;
	GpuTimer timer;

	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	double2 *d_subgrid;
	double2 *d_output_visibilities;
	double *d_gcf_uv_kernel;
	double *d_gcf_w_kernel;
	double *d_u_vis_pos;
	double *d_v_vis_pos;
	double *d_w_vis_pos;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_subgrid,             subgrid_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output_visibilities, output_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_gcf_uv_kernel,       uv_kernel_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_gcf_w_kernel,        w_kernel_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_u_vis_pos,           visibilities_position_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_v_vis_pos,           visibilities_position_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_w_vis_pos,           visibilities_position_size_in_bytes));
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> FFT calculation
		//-----> Copy chunk of input data to a device
		checkCudaErrors(cudaMemcpy(d_subgrid,       h_subgrid,       subgrid_size_in_bytes,               cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_gcf_uv_kernel, h_gcf_uv_kernel, uv_kernel_size_in_bytes,             cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_gcf_w_kernel,  h_gcf_w_kernel,  w_kernel_size_in_bytes,              cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_u_vis_pos,     h_u_vis_pos,     visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v_vis_pos,     h_v_vis_pos,     visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_w_vis_pos,     h_w_vis_pos,     visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		
		//-----> Compute LEHRMS
		for(int f=0; f<nRuns; f++){
			SKA_degrid_benchmark(
				d_output_visibilities, 
				d_gcf_uv_kernel, 
				uv_kernel_size, 
				uv_kernel_stride, 
				uv_kernel_oversampling, 
				d_gcf_w_kernel, 
				w_kernel_size, 
				w_kernel_stride, 
				w_kernel_oversampling, 
				d_subgrid, 
				grid_z, 
				grid_y, 
				grid_x, 
				d_u_vis_pos, 
				d_v_vis_pos, 
				d_w_vis_pos, 
				nVisibilities, 
				nSubgrids, 
				theta,
				wstep,
				&exec_time
			);
		}
		exec_time = exec_time/((float) nRuns);
		
		*execution_time = exec_time;
		
		checkCudaErrors(cudaGetLastError());
		
		//-----> Copy chunk of output data to host
		checkCudaErrors(cudaMemcpy(h_output_visibilities, d_output_visibilities, output_size_in_bytes, cudaMemcpyDeviceToHost));
	//------------------------------------<
		
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_subgrid));
	checkCudaErrors(cudaFree(d_output_visibilities));
	checkCudaErrors(cudaFree(d_gcf_uv_kernel));
	checkCudaErrors(cudaFree(d_gcf_w_kernel));
	checkCudaErrors(cudaFree(d_u_vis_pos));
	checkCudaErrors(cudaFree(d_v_vis_pos));
	checkCudaErrors(cudaFree(d_w_vis_pos));
	
	return(0);
}

