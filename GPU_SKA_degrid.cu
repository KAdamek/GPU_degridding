#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "params.h"
#include "timer.h"
#include "utils_cuda.h"

#include <iostream>
#include <fstream>
#include <iomanip> 

using namespace std;

class SKA_degrid_params {
public:
	static const int warp = 32;
};

class SKA_degrid_8_8_4 : public SKA_degrid_params {
public:
	static const int uv_kernel_size = 8;
	static const int uv_kernel_stride = 8;
	static const int uv_oversample = 16384;
	static const int w_kernel_size = 4;
	static const int w_kernel_stride = 4;
	static const int w_oversample = 16384;
	static const int grid_x = 512;
	static const int grid_y = 512;
	static const int grid_z = 4;
};

class SKA_degrid_8_8_4_1 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 1;
};

class SKA_degrid_8_8_4_5 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 5;
};

class SKA_degrid_8_8_4_7 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 7;
};

class SKA_degrid_8_8_4_10 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 10;
};

class SKA_degrid_8_8_4_15 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 15;
};

class SKA_degrid_8_8_4_20 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 20;
};

class SKA_degrid_8_8_4_30 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 30;
};

class SKA_degrid_8_8_4_32 : public SKA_degrid_8_8_4 {
public:
	static const int nvis_per_block = 32;
};

#define HALF_WARP 16
#define WARP 32


// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************

template<class const_params>
__inline__ __device__ void calculate_coordinates(
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
	double ox = x*const_params::uv_oversample;
	int iox = __double2int_rn(ox); // round to nearest
    iox += (const_params::grid_x / 2 + 1) * const_params::uv_oversample - 1;
    int home_x = iox / const_params::uv_oversample;
    int frac_x = const_params::uv_oversample - 1 - (iox % const_params::uv_oversample);
	
	// y coordinate
	double y = theta*v;
	double oy = y*const_params::uv_oversample;
	int ioy = __double2int_rn(oy);
    ioy += (const_params::grid_x / 2 + 1) * const_params::uv_oversample - 1;
    int home_y = ioy / const_params::uv_oversample;
    int frac_y = const_params::uv_oversample - 1 - (ioy % const_params::uv_oversample);
	
	// w coordinate
	double z = 1.0 + w/wstep;
	double oz = z*const_params::w_oversample;
	int ioz = __double2int_rn(oz);
    ioz += const_params::w_oversample - 1;
    int frac_z = const_params::w_oversample - 1 - (ioz % const_params::w_oversample);
	
    *grid_offset = (home_y-const_params::uv_kernel_size/2)*const_params::grid_x + (home_x-const_params::uv_kernel_size/2);
    *sub_offset_x = const_params::uv_kernel_stride * frac_x;
    *sub_offset_y = const_params::uv_kernel_stride * frac_y;
    *sub_offset_z = const_params::w_kernel_stride * frac_z;
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
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	int active_nVisibilities = nVisibilities[blockIdx.y + 1] - nVisibilities[blockIdx.y];
	if(blockIdx.x >= active_nVisibilities) {
		return;
	}
	
	int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
	calculate_coordinates<const_params>(
		theta, wstep, 
		d_u_vis_pos[nVisibilities[blockIdx.y] + blockIdx.x], 
		d_v_vis_pos[nVisibilities[blockIdx.y] + blockIdx.x], 
		d_w_vis_pos[nVisibilities[blockIdx.y] + blockIdx.x],
		&grid_offset, 
		&sub_offset_x, &sub_offset_y, &sub_offset_z
	);
	
	double2 vis;
	vis.x = 0; vis.y = 0;
	int local_x = (threadIdx.x&7);
	int local_y = (threadIdx.x>>3);
	
	for (int z = 0; z < const_params::w_kernel_size; z++) {
		double2 grid_value = d_subgrid[blockIdx.y*const_params::grid_z*const_params::grid_y*const_params::grid_x + z*const_params::grid_x*const_params::grid_y + grid_offset + local_y*const_params::grid_y + local_x];
		
		//vis.x += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.x;
		//vis.y += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.y;
		vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
		vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
	}
	vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
	vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
	
	// NOTE: Reduction checked
	s_local[threadIdx.x] = vis;
	__syncthreads();
	double2 sum;
	sum = Reduce_SM(s_local);
	Reduce_WARP(&sum);	
	__syncthreads();
	
	if(threadIdx.x==0) {
		d_output_visibilities[nVisibilities[blockIdx.y] + blockIdx.x] = sum;
	}
}

template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk1_for_dynamic(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep,
		int block_y
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	
	int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
	calculate_coordinates<const_params>(
		theta, wstep, 
		d_u_vis_pos[nVisibilities[block_y] + blockIdx.x], 
		d_v_vis_pos[nVisibilities[block_y] + blockIdx.x], 
		d_w_vis_pos[nVisibilities[block_y] + blockIdx.x],
		&grid_offset, 
		&sub_offset_x, &sub_offset_y, &sub_offset_z
	);
	
	double2 vis;
	vis.x = 0; vis.y = 0;
	int local_x = (threadIdx.x&7);
	int local_y = (threadIdx.x>>3);
	
	for (int z = 0; z < const_params::w_kernel_size; z++) {
		double2 grid_value = d_subgrid[block_y*const_params::grid_z*const_params::grid_y*const_params::grid_x + z*const_params::grid_x*const_params::grid_y + grid_offset + local_y*const_params::grid_y + local_x];
		
		//vis.x += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.x;
		//vis.y += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.y;
		vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
		vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
	}
	vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
	vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
	
	// NOTE: Reduction checked
	s_local[threadIdx.x] = vis;
	__syncthreads();
	double2 sum;
	sum = Reduce_SM(s_local);
	Reduce_WARP(&sum);	
	__syncthreads();
	
	if(threadIdx.x==0) {
		d_output_visibilities[nVisibilities[block_y] + blockIdx.x] = sum;
	}
}

template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk1_dynamic(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *d_nVisibilities, 
		double theta,
		double wstep
	){
	// each thread represent one subgrid
	int active_nVisibilities = d_nVisibilities[threadIdx.x + 1] - d_nVisibilities[threadIdx.x];
	dim3 gridSize(active_nVisibilities, 1, 1);
	dim3 blockSize(const_params::uv_kernel_stride*const_params::uv_kernel_stride, 1, 1);
	int shared_mem = const_params::uv_kernel_stride*const_params::uv_kernel_stride*sizeof(double2);
	GPU_SKA_degrid_kernel_mk1_for_dynamic<const_params><<< gridSize , blockSize, shared_mem >>>(
		d_output_visibilities, 
		d_gcf_uv_kernel,
		d_gcf_w_kernel,
		d_subgrid,
		d_u_vis_pos, 
		d_v_vis_pos, 
		d_w_vis_pos, 
		d_nVisibilities, 
		theta, 
		wstep,
		threadIdx.x
	);
	
}


template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk2(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	int active_nVisibilities = nVisibilities[blockIdx.y + 1] - nVisibilities[blockIdx.y];
	if(blockIdx.x*const_params::nvis_per_block >= active_nVisibilities) {
		return;
	}
	
	int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
	for(int f=0; f<const_params::nvis_per_block; f++){
		if( blockIdx.x*const_params::nvis_per_block + f < active_nVisibilities ){
			int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + f;
			calculate_coordinates<const_params>(
				theta, wstep, 
				d_u_vis_pos[pos], 
				d_v_vis_pos[pos], 
				d_w_vis_pos[pos],
				&grid_offset, 
				&sub_offset_x, &sub_offset_y, &sub_offset_z
			);
			
			double2 vis;
			vis.x = 0; vis.y = 0;
			int local_x = (threadIdx.x&7);
			int local_y = (threadIdx.x>>3);
			grid_offset = blockIdx.y*const_params::grid_z*const_params::grid_y*const_params::grid_x + grid_offset + local_y*const_params::grid_y + local_x;
			
			for (int z = 0; z < const_params::w_kernel_size; z++) {
				double2 grid_value = d_subgrid[grid_offset + z*const_params::grid_x*const_params::grid_y];
				
				//vis.x += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.x;
				//vis.y += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.y;
				vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
				vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
			}
			vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			
			// NOTE: Reduction checked
			s_local[threadIdx.x] = vis;
			__syncthreads();
			double2 sum;
			sum = Reduce_SM(s_local);
			Reduce_WARP(&sum);	
			__syncthreads();
			
			if(threadIdx.x==0) {
				d_output_visibilities[pos] = sum;
			}
		}
	}
}


template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk3(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	__shared__ int s_coordinates[const_params::nvis_per_block*4];
	int active_nVisibilities = nVisibilities[blockIdx.y + 1] - nVisibilities[blockIdx.y];
	if(blockIdx.x*const_params::nvis_per_block >= active_nVisibilities) {
		return;
	}
	
	// precalculate coordinates;
	if(threadIdx.x < const_params::nvis_per_block && (blockIdx.x*const_params::nvis_per_block + threadIdx.x) < active_nVisibilities){
		int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
		int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		calculate_coordinates<const_params>(
			theta, wstep, 
			d_u_vis_pos[pos], 
			d_v_vis_pos[pos], 
			d_w_vis_pos[pos],
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);
		s_coordinates[threadIdx.x                   ] = grid_offset;
		s_coordinates[threadIdx.x + const_params::nvis_per_block  ] = sub_offset_x;
		s_coordinates[threadIdx.x + 2*const_params::nvis_per_block] = sub_offset_y;
		s_coordinates[threadIdx.x + 3*const_params::nvis_per_block] = sub_offset_z;
	}
	__syncthreads();
	
	
	for(int f=0; f<const_params::nvis_per_block; f++){
		if( blockIdx.x*const_params::nvis_per_block + f < active_nVisibilities ){
			int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
			grid_offset  = s_coordinates[f];
			sub_offset_x = s_coordinates[f + const_params::nvis_per_block  ];
			sub_offset_y = s_coordinates[f + 2*const_params::nvis_per_block];
			sub_offset_z = s_coordinates[f + 3*const_params::nvis_per_block];
			
			double2 vis;
			vis.x = 0; vis.y = 0;
			int local_x = (threadIdx.x&7);
			int local_y = (threadIdx.x>>3);
			grid_offset = blockIdx.y*const_params::grid_z*const_params::grid_y*const_params::grid_x + grid_offset + local_y*const_params::grid_y + local_x;
			
			for (int z = 0; z < const_params::w_kernel_size; z++) {
				double2 grid_value = d_subgrid[grid_offset + z*const_params::grid_x*const_params::grid_y];
				
				vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
				vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
			}
			vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			
			// NOTE: Reduction checked
			s_local[threadIdx.x] = vis;
			__syncthreads();
			double2 sum;
			sum = Reduce_SM(s_local);
			Reduce_WARP(&sum);	
			__syncthreads();
			
			if(threadIdx.x==0) {
				int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + f;
				d_output_visibilities[pos] = sum;
			}
		}
	}
}


template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk3_for_dynamic(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep,
		int block_y
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	int active_nVisibilities = nVisibilities[block_y + 1] - nVisibilities[block_y];
	__shared__ int s_coordinates[const_params::nvis_per_block*4];
	if(blockIdx.x*const_params::nvis_per_block >= active_nVisibilities) {
		return;
	}
	
	// precalculate coordinates;
	if(threadIdx.x < const_params::nvis_per_block && (blockIdx.x*const_params::nvis_per_block + threadIdx.x) < active_nVisibilities){
		int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
		int pos = nVisibilities[block_y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		calculate_coordinates<const_params>(
			theta, wstep, 
			d_u_vis_pos[pos], 
			d_v_vis_pos[pos], 
			d_w_vis_pos[pos],
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);
		s_coordinates[threadIdx.x                   ] = grid_offset;
		s_coordinates[threadIdx.x + const_params::nvis_per_block  ] = sub_offset_x;
		s_coordinates[threadIdx.x + 2*const_params::nvis_per_block] = sub_offset_y;
		s_coordinates[threadIdx.x + 3*const_params::nvis_per_block] = sub_offset_z;
	}
	__syncthreads();
	
	
	for(int f=0; f<const_params::nvis_per_block; f++){
		if( blockIdx.x*const_params::nvis_per_block + f < active_nVisibilities ){
			int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
			grid_offset  = s_coordinates[f];
			sub_offset_x = s_coordinates[f + const_params::nvis_per_block  ];
			sub_offset_y = s_coordinates[f + 2*const_params::nvis_per_block];
			sub_offset_z = s_coordinates[f + 3*const_params::nvis_per_block];
			
			double2 vis;
			vis.x = 0; vis.y = 0;
			int local_x = (threadIdx.x&7);
			int local_y = (threadIdx.x>>3);
			grid_offset = block_y*const_params::grid_z*const_params::grid_y*const_params::grid_x + grid_offset + local_y*const_params::grid_y + local_x;
			
			for (int z = 0; z < const_params::w_kernel_size; z++) {
				double2 grid_value = d_subgrid[grid_offset + z*const_params::grid_x*const_params::grid_y];
				
				//vis.x += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.x;
				//vis.y += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.y;
				vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
				vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
			}
			vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			
			// NOTE: Reduction checked
			s_local[threadIdx.x] = vis;
			__syncthreads();
			double2 sum;
			sum = Reduce_SM(s_local);
			Reduce_WARP(&sum);	
			__syncthreads();
			
			if(threadIdx.x==0) {
				int pos = nVisibilities[block_y] + blockIdx.x*const_params::nvis_per_block + f;
				d_output_visibilities[pos] = sum;
			}
		}
	}
}


template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk3_dynamic(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *d_nVisibilities, 
		double theta,
		double wstep
	){
	// each thread represent one subgrid
	int active_nVisibilities = d_nVisibilities[threadIdx.x + 1] - d_nVisibilities[threadIdx.x];
	dim3 gridSize((active_nVisibilities + const_params::nvis_per_block - 1)/const_params::nvis_per_block, 1, 1);
	dim3 blockSize(const_params::uv_kernel_stride*const_params::uv_kernel_stride, 1, 1);
	int shared_mem = const_params::uv_kernel_stride*const_params::uv_kernel_stride*sizeof(double2);
	GPU_SKA_degrid_kernel_mk3_for_dynamic<const_params><<< gridSize , blockSize, shared_mem >>>(
		d_output_visibilities, 
		d_gcf_uv_kernel,
		d_gcf_w_kernel,
		d_subgrid,
		d_u_vis_pos, 
		d_v_vis_pos, 
		d_w_vis_pos, 
		d_nVisibilities, 
		theta, 
		wstep,
		threadIdx.x
	);
	
}

template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk5(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	int active_nVisibilities = nVisibilities[blockIdx.y + 1] - nVisibilities[blockIdx.y];
	__shared__ int s_coordinates[const_params::nvis_per_block*4];
	if(blockIdx.x*const_params::nvis_per_block >= active_nVisibilities) {
		return;
	}
	
	// precalculate coordinates;
	if(threadIdx.x < const_params::nvis_per_block && (blockIdx.x*const_params::nvis_per_block + threadIdx.x) < active_nVisibilities){
		int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
		int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		calculate_coordinates<const_params>(
			theta, wstep, 
			d_vis_pos[3*pos], 
			d_vis_pos[3*pos + 1], 
			d_vis_pos[3*pos + 2],
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);
		s_coordinates[threadIdx.x                   ] = grid_offset;
		s_coordinates[threadIdx.x + const_params::nvis_per_block  ] = sub_offset_x;
		s_coordinates[threadIdx.x + 2*const_params::nvis_per_block] = sub_offset_y;
		s_coordinates[threadIdx.x + 3*const_params::nvis_per_block] = sub_offset_z;
	}
	__syncthreads();
	
	
	for(int f=0; f<const_params::nvis_per_block; f++){
		if( blockIdx.x*const_params::nvis_per_block + f < active_nVisibilities ){
			int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
			grid_offset  = s_coordinates[f];
			sub_offset_x = s_coordinates[f + const_params::nvis_per_block  ];
			sub_offset_y = s_coordinates[f + 2*const_params::nvis_per_block];
			sub_offset_z = s_coordinates[f + 3*const_params::nvis_per_block];
			
			double2 vis;
			vis.x = 0; vis.y = 0;
			int local_x = (threadIdx.x&7);
			int local_y = (threadIdx.x>>3);
			grid_offset = blockIdx.y*const_params::grid_z*const_params::grid_y*const_params::grid_x + grid_offset + local_y*const_params::grid_y + local_x;
			
			for (int z = 0; z < const_params::w_kernel_size; z++) {
				double2 grid_value = d_subgrid[grid_offset + z*const_params::grid_x*const_params::grid_y];
				
				//vis.x += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.x;
				//vis.y += d_gcf_w_kernel[sub_offset_z + z]*d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x]*grid_value.y;
				vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
				vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
			}
			vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			
			// NOTE: Reduction checked
			s_local[threadIdx.x] = vis;
			__syncthreads();
			double2 sum;
			sum = Reduce_SM(s_local);
			Reduce_WARP(&sum);	
			__syncthreads();
			
			if(threadIdx.x==0) {
				int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + f;
				d_output_visibilities[pos] = sum;
			}
		}
	}
}

template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk7(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::uv_kernel_stride];
	__shared__ double2 s_final_visibilities[const_params::nvis_per_block];
	int active_nVisibilities = nVisibilities[blockIdx.y + 1] - nVisibilities[blockIdx.y];
	__shared__ int s_coordinates[const_params::nvis_per_block*4];
	if(blockIdx.x*const_params::nvis_per_block >= active_nVisibilities) {
		return;
	}
	
	// precalculate coordinates;
	if(threadIdx.x < const_params::nvis_per_block && (blockIdx.x*const_params::nvis_per_block + threadIdx.x) < active_nVisibilities){
		int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
		int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		calculate_coordinates<const_params>(
			theta, wstep, 
			d_u_vis_pos[pos], 
			d_v_vis_pos[pos], 
			d_w_vis_pos[pos],
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);
		s_coordinates[threadIdx.x                   ] = grid_offset;
		s_coordinates[threadIdx.x + const_params::nvis_per_block  ] = sub_offset_x;
		s_coordinates[threadIdx.x + 2*const_params::nvis_per_block] = sub_offset_y;
		s_coordinates[threadIdx.x + 3*const_params::nvis_per_block] = sub_offset_z;
	}
	__syncthreads();
	
	
	for(int f=0; f<const_params::nvis_per_block; f++){
		if( blockIdx.x*const_params::nvis_per_block + f < active_nVisibilities ){
			int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
			grid_offset  = s_coordinates[f];
			sub_offset_x = s_coordinates[f + const_params::nvis_per_block  ];
			sub_offset_y = s_coordinates[f + 2*const_params::nvis_per_block];
			sub_offset_z = s_coordinates[f + 3*const_params::nvis_per_block];
			
			double2 vis;
			vis.x = 0; vis.y = 0;
			int local_x = (threadIdx.x&7);
			int local_y = (threadIdx.x>>3);
			grid_offset = blockIdx.y*const_params::grid_z*const_params::grid_y*const_params::grid_x + grid_offset + local_y*const_params::grid_y + local_x;
			
			for (int z = 0; z < const_params::w_kernel_size; z++) {
				double2 grid_value = d_subgrid[grid_offset + z*const_params::grid_x*const_params::grid_y];
				
				vis.x += d_gcf_w_kernel[sub_offset_z + z]*grid_value.x;
				vis.y += d_gcf_w_kernel[sub_offset_z + z]*grid_value.y;
			}
			vis.x *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			vis.y *= d_gcf_uv_kernel[sub_offset_y + local_y]*d_gcf_uv_kernel[sub_offset_x + local_x];
			
			// NOTE: Reduction checked
			//s_local[threadIdx.x] = vis;
			//__syncthreads();
			//double2 sum;
			//sum = Reduce_SM(s_local);
			//Reduce_WARP(&sum);
			//if(threadIdx.x==0) s_final_visibilities[f] = sum;
			//__syncthreads();
			
			// NOTE: Reduction checked
			s_local[threadIdx.x] = vis;
			__syncthreads();
			double2 sum;
			if(threadIdx.x<const_params::warp){
				sum.x = s_local[threadIdx.x].x + s_local[const_params::warp + threadIdx.x].x;
				sum.y = s_local[threadIdx.x].y + s_local[const_params::warp + threadIdx.x].y;
				Reduce_WARP(&sum);
				if(threadIdx.x==0) s_final_visibilities[f] = sum;
			}
			__syncthreads();
		}
	}
	
	if(threadIdx.x<const_params::nvis_per_block && blockIdx.x*const_params::nvis_per_block + threadIdx.x < active_nVisibilities) {
		int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		d_output_visibilities[pos] = s_final_visibilities[threadIdx.x];
	}
}

template<class const_params>
__global__ void GPU_SKA_degrid_kernel_mk8(
		double2 *d_output_visibilities, 
		double *d_gcf_uv_kernel, 
		double *d_gcf_w_kernel, 
		double2 *d_subgrid, 
		double *d_u_vis_pos, 
		double *d_v_vis_pos, 
		double *d_w_vis_pos, 
		int *nVisibilities, 
		double theta,
		double wstep
	){
	__shared__ double2 s_local[const_params::uv_kernel_stride*const_params::w_kernel_stride];
	__shared__ double2 s_final_visibilities[const_params::nvis_per_block];
	int active_nVisibilities = nVisibilities[blockIdx.y + 1] - nVisibilities[blockIdx.y];
	__shared__ int s_coordinates[const_params::nvis_per_block*4];
	if(blockIdx.x*const_params::nvis_per_block >= active_nVisibilities) {
		return;
	}
	
	// precalculate coordinates;
	if(threadIdx.x < const_params::nvis_per_block && (blockIdx.x*const_params::nvis_per_block + threadIdx.x) < active_nVisibilities){
		int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
		int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		calculate_coordinates<const_params>(
			theta, wstep, 
			d_u_vis_pos[pos], 
			d_v_vis_pos[pos], 
			d_w_vis_pos[pos],
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);
		s_coordinates[threadIdx.x                   ] = grid_offset;
		s_coordinates[threadIdx.x + const_params::nvis_per_block  ] = sub_offset_x;
		s_coordinates[threadIdx.x + 2*const_params::nvis_per_block] = sub_offset_y;
		s_coordinates[threadIdx.x + 3*const_params::nvis_per_block] = sub_offset_z;
	}
	__syncthreads();
	
	
	for(int f=0; f<const_params::nvis_per_block; f++){
		if( blockIdx.x*const_params::nvis_per_block + f < active_nVisibilities ){
			int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
			grid_offset  = s_coordinates[f];
			sub_offset_x = s_coordinates[f + const_params::nvis_per_block  ];
			sub_offset_y = s_coordinates[f + 2*const_params::nvis_per_block];
			sub_offset_z = s_coordinates[f + 3*const_params::nvis_per_block];
			
			double2 vis;
			vis.x = 0; vis.y = 0;
			int local_x = (threadIdx.x&7);
			int local_z = (threadIdx.x>>3);
			grid_offset = blockIdx.y*const_params::grid_z*const_params::grid_y*const_params::grid_x + grid_offset + local_z*const_params::grid_x*const_params::grid_y + local_x;
			
			for (int v = 0; v < const_params::uv_kernel_size; v++) {
				double2 grid_value = d_subgrid[grid_offset + v*const_params::grid_x];
				
				vis.x += d_gcf_uv_kernel[sub_offset_y + v]*grid_value.x;
				vis.y += d_gcf_uv_kernel[sub_offset_y + v]*grid_value.y;
			}
			vis.x *= d_gcf_w_kernel[sub_offset_z + local_z]*d_gcf_uv_kernel[sub_offset_x + local_x];
			vis.y *= d_gcf_w_kernel[sub_offset_z + local_z]*d_gcf_uv_kernel[sub_offset_x + local_x];
			
			Reduce_WARP(&vis);
			if(threadIdx.x==0) s_final_visibilities[f] = vis;
		}
	}
	
	if(threadIdx.x<const_params::nvis_per_block && blockIdx.x*const_params::nvis_per_block + threadIdx.x < active_nVisibilities) {
		int pos = nVisibilities[blockIdx.y] + blockIdx.x*const_params::nvis_per_block + threadIdx.x;
		d_output_visibilities[pos] = s_final_visibilities[threadIdx.x];
	}
}


void SKA_init(){
	//---------> Specific nVidia stuff
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
}


int SKA_degrid_benchmark_mk1(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize(max_nVisibilities, nSubgrids, 1);
	dim3 blockSize(uv_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = uv_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Degridding using basic GPU kernel
	SKA_init();
	GPU_SKA_degrid_kernel_mk1<SKA_degrid_8_8_4><<< gridSize , blockSize, shared_mem >>>(
		d_output_visibilities, 
		d_gcf_uv_kernel,
		d_gcf_w_kernel,
		d_subgrid,
		d_u_vis_pos, 
		d_v_vis_pos, 
		d_w_vis_pos, 
		d_nVisibilities, 
		theta, 
		wstep
	);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk1_dynamic(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(nSubgrids, 1, 1);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Degridding using basic GPU kernel with dynamic parallelism
	SKA_init();
	GPU_SKA_degrid_kernel_mk1_dynamic<SKA_degrid_8_8_4><<< gridSize , blockSize >>>(
		d_output_visibilities, 
		d_gcf_uv_kernel,
		d_gcf_w_kernel,
		d_subgrid,
		d_u_vis_pos, 
		d_v_vis_pos, 
		d_w_vis_pos, 
		d_nVisibilities, 
		theta, 
		wstep
	);
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk2(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize((int) ((max_nVisibilities + vis_per_block - 1)/vis_per_block), nSubgrids, 1);
	dim3 blockSize(uv_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = uv_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Degridding using batched GPU kernel
	SKA_init();
	switch(vis_per_block) {
		case 1:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_1><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 5:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_5><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 7:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_7><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 10:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_10><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 15:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_15><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 20:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_20><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 30:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_30><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 32:
			GPU_SKA_degrid_kernel_mk2<SKA_degrid_8_8_4_32><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		default:
			printf("Wrong number of visibilities processed per threadblock.\n");
			break;
	}
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk3(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize((int) ((max_nVisibilities + vis_per_block - 1)/vis_per_block), nSubgrids, 1);
	dim3 blockSize(uv_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = uv_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Degridding using batched GPU kernel with precomputed visibilities
	SKA_init();
	switch(vis_per_block) {
		case 1:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_1><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 5:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_5><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 7:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_7><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 10:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_10><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 15:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_15><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 20:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_20><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 30:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_30><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 32:
			GPU_SKA_degrid_kernel_mk3<SKA_degrid_8_8_4_32><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		default:
			printf("Wrong number of visibilities processed per threadblock.\n");
			break;
	}
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk3_dynamic(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(nSubgrids, 1, 1);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Degridding using batched GPU kernel with precomputed coordinates and dynamic parallelism
	SKA_init();
	switch(vis_per_block) {
		case 1:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_1><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 5:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_5><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 7:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_7><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 10:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_10><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 15:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_15><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 20:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_20><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 30:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_30><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 32:
			GPU_SKA_degrid_kernel_mk3_dynamic<SKA_degrid_8_8_4_32><<< gridSize , blockSize >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		default:
			printf("Wrong number of visibilities processed per threadblock.\n");
			break;
	}
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk5(
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
		double *d_vis_pos, 
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize((int) ((max_nVisibilities + vis_per_block - 1)/vis_per_block), nSubgrids, 1);
	dim3 blockSize(uv_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = uv_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Pulse detection FIR
	SKA_init();
	switch(vis_per_block) {
		case 1:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_1><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 5:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_5><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 7:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_7><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 10:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_10><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 15:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_15><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 20:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_20><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 30:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_30><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 32:
			GPU_SKA_degrid_kernel_mk5<SKA_degrid_8_8_4_32><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_vis_pos, d_nVisibilities, theta, wstep);
			break;
		default:
			printf("Wrong number of visibilities processed per threadblock.\n");
			break;
	}
	
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk7(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize((int) ((max_nVisibilities + vis_per_block - 1)/vis_per_block), nSubgrids, 1);
	dim3 blockSize(uv_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = uv_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Pulse detection FIR
	SKA_init();
	switch(vis_per_block) {
		case 1:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_1><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 5:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_5><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 7:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_7><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 10:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_10><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 15:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_15><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 20:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_20><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 30:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_30><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 32:
			GPU_SKA_degrid_kernel_mk7<SKA_degrid_8_8_4_32><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		default:
			printf("Wrong number of visibilities processed per threadblock.\n");
			break;
	}
	
	
	timer.Stop();
	*exec_time += timer.Elapsed();
	// --------> Measured part
	// ----------------------------------------------->
	return(0);
}

int SKA_degrid_benchmark_mk8(
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
		int *d_nVisibilities,
		int max_nVisibilities,
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		double *exec_time
	){
	GpuTimer timer;
	
	//---------> Task specific
	dim3 gridSize((int) ((max_nVisibilities + vis_per_block - 1)/vis_per_block), nSubgrids, 1);
	dim3 blockSize(w_kernel_stride*uv_kernel_stride, 1, 1);
	size_t shared_mem = w_kernel_stride*uv_kernel_stride*sizeof(double2);
	
	if(DEBUG) printf("Grid  settings: x:%d; y:%d; z:%d;\n", gridSize.x, gridSize.y, gridSize.z);
	if(DEBUG) printf("Block settings: x:%d; y:%d; z:%d;\n", blockSize.x, blockSize.y, blockSize.z);
	
	// ----------------------------------------------->
	// --------> Measured part
	timer.Start();
	
	//---------> Pulse detection FIR
	SKA_init();
	switch(vis_per_block) {
		case 1:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_1><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid,d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 5:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_5><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 7:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_7><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 10:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_10><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 15:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_15><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 20:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_20><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 30:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_30><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		case 32:
			GPU_SKA_degrid_kernel_mk8<SKA_degrid_8_8_4_32><<< gridSize , blockSize, shared_mem >>>(d_output_visibilities, d_gcf_uv_kernel, d_gcf_w_kernel, d_subgrid, d_u_vis_pos, d_v_vis_pos, d_w_vis_pos, d_nVisibilities, theta, wstep);
			break;
		default:
			printf("Wrong number of visibilities processed per threadblock.\n");
			break;
	}
	
	
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
		size_t total_nVisibilities,
		int *h_nVisibilities, 
		int max_nVisibilities, 
		int nSubgrids,
		double theta,
		double wstep,
		int vis_per_block,
		int kernel_type,
		int nRuns,
		int device,
		double *execution_time
	) {
	//---------> Initial nVidia stuff
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount) checkCudaErrors(cudaSetDevice(device));
	else { printf("Wrong device!\n"); exit(1); }
	
	//if(kernel_type==2) return(1); // No performance benefit
	//if(kernel_type==5) return(1); // No performance benefit
	
	size_t uv_kernel_size_in_bytes             = uv_kernel_stride*uv_kernel_oversampling*sizeof(double);
	size_t w_kernel_size_in_bytes              = w_kernel_stride*w_kernel_oversampling*sizeof(double);
	size_t subgrid_size_in_bytes               = grid_z*grid_y*grid_x*sizeof(double2)*nSubgrids;
	size_t visibilities_position_size_in_bytes = total_nVisibilities*sizeof(double);
	size_t output_size_in_bytes                = total_nVisibilities*sizeof(double2);
	size_t total_memory_required_in_bytes      = uv_kernel_size_in_bytes + w_kernel_size_in_bytes + subgrid_size_in_bytes + visibilities_position_size_in_bytes + output_size_in_bytes;
	
	printf("uv_kernel_size_in_bytes            : %zu;\n", uv_kernel_size_in_bytes);
	printf("w_kernel_size_in_bytes             : %zu;\n", w_kernel_size_in_bytes);
	printf("subgrid_size_in_bytes              : %zu;\n", subgrid_size_in_bytes);
	printf("visibilities_position_size_in_bytes: %zu;\n", visibilities_position_size_in_bytes);
	printf("output_size_in_bytes               : %zu;\n", output_size_in_bytes);
	printf("total_memory_required_in_bytes     : %zu;\n", total_memory_required_in_bytes);
	
	
	//---------> Checking memory
	if(check_memory(total_memory_required_in_bytes, 1.0)!=0) return(1);
	
	//---------> Measurements
	double exec_time = 0;
	GpuTimer timer;
	
	//---------> Measurements
	printf("Preparing one visibility array...\n");
	double *h_vis_pos;
	h_vis_pos = new double[3*total_nVisibilities];
	for(size_t f=0; f<total_nVisibilities; f++){
		h_vis_pos[3*f + 0] = h_u_vis_pos[f];
		h_vis_pos[3*f + 1] = h_v_vis_pos[f];
		h_vis_pos[3*f + 2] = h_w_vis_pos[f];
	}

	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	double2 *d_subgrid;
	double2 *d_output_visibilities;
	double *d_gcf_uv_kernel;
	double *d_gcf_w_kernel;
	double *d_u_vis_pos;
	double *d_v_vis_pos;
	double *d_w_vis_pos;
	double *d_vis_pos;
	int *d_nVisibilities;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_subgrid,             subgrid_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output_visibilities, output_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_gcf_uv_kernel,       uv_kernel_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_gcf_w_kernel,        w_kernel_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_u_vis_pos,           visibilities_position_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_v_vis_pos,           visibilities_position_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_w_vis_pos,           visibilities_position_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_vis_pos,             3*visibilities_position_size_in_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_nVisibilities,       (nSubgrids+1)*sizeof(int)));
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> degridding calculation
		//-----> Copy chunk of input data to a device
		timer.Start();
		checkCudaErrors(cudaMemcpy(d_subgrid,       h_subgrid,       subgrid_size_in_bytes,               cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_gcf_uv_kernel, h_gcf_uv_kernel, uv_kernel_size_in_bytes,             cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_gcf_w_kernel,  h_gcf_w_kernel,  w_kernel_size_in_bytes,              cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_u_vis_pos,     h_u_vis_pos,     visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v_vis_pos,     h_v_vis_pos,     visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_w_vis_pos,     h_w_vis_pos,     visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vis_pos,       h_vis_pos,       3*visibilities_position_size_in_bytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_nVisibilities, h_nVisibilities, (nSubgrids+1)*sizeof(int), cudaMemcpyHostToDevice));
		timer.Stop();
		size_t total_size = subgrid_size_in_bytes + uv_kernel_size_in_bytes + w_kernel_size_in_bytes + 3*visibilities_position_size_in_bytes;
		printf("Time to copy data to the device: %f; PCIe bandwidth: %f GB/s;\n", timer.Elapsed(), 1000.0*(total_size/timer.Elapsed())/(1024.0*1024.0*1024.0));
		delete[] h_vis_pos;
		
		//-----> Compute degridding
		
		if(kernel_type==1 && vis_per_block==1){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				//SKA_degrid_benchmark_mk1(
				SKA_degrid_benchmark_mk1(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using basic finished in %fms\n", exec_time);
		}
		
		
		if(kernel_type==2){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				SKA_degrid_benchmark_mk2(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					vis_per_block,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using batched GPU kernel finished in %fms\n", exec_time);
		}
		
		
		if(kernel_type==3){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				SKA_degrid_benchmark_mk3(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					vis_per_block,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using batched GPU kernel with precalculated coordinates finished in %fms\n", exec_time);
		}
		
		
		if(kernel_type==4 && vis_per_block==1){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				//SKA_degrid_benchmark_mk1(
				SKA_degrid_benchmark_mk1_dynamic(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using basic GPU kernel with dynamic parallelism finished in %fms\n", exec_time);
		}


		if(kernel_type==5){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				//SKA_degrid_benchmark_mk1(
				SKA_degrid_benchmark_mk5(
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
					d_vis_pos, 
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					vis_per_block,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using batched GPU kernel with single u,v,w array finished in %fms\n", exec_time);
		}

		
		if(kernel_type==7){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				SKA_degrid_benchmark_mk7(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					vis_per_block,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using batched GPU kernel with single u,v,w array finished in %fms\n", exec_time);
		}
		
		
		if(kernel_type==8){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				SKA_degrid_benchmark_mk8(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					vis_per_block,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using batched GPU kernel with single u,v,w array finished in %fms\n", exec_time);
		}
		
		
		if(kernel_type==6){
			exec_time = 0;
			for(int f=0; f<nRuns; f++){
				//SKA_degrid_benchmark_mk1(
				SKA_degrid_benchmark_mk3_dynamic(
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
					d_nVisibilities,
					max_nVisibilities,
					nSubgrids, 
					theta,
					wstep,
					vis_per_block,
					&exec_time
				);
			}
			exec_time = exec_time/((float) nRuns);
			*execution_time = exec_time;
			printf("Degridding using batched GPU kernel with precalculated coordinates with dynamic parallelism finished in %fms\n", exec_time);
		}
		
		
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
	checkCudaErrors(cudaFree(d_vis_pos));
	checkCudaErrors(cudaFree(d_nVisibilities));
	
	return(0);
}

