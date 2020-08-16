#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "debug.h"
#include "params.h"
#include "results.h"

#define RANDOMDATA



//-----------------------------------------------
//---------- Data checks
double max_error = 1.0e-4;

float get_error(float A, float B){
	float error, div_error=10000, per_error=10000, order=0;
	int power;
	if(A<0) A = -A;
	if(B<0) B = -B;
	
	if (A>B) {
		div_error = A-B;
		if(B>10){
			power = (int) log10(B);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	else {
		div_error = B-A;
		if(A>10){
			power = (int) log10(A);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	
	if(div_error<per_error) error = div_error;
	else error = per_error;
	return(error);
}

size_t Compare_data(int *CPU_data, int *GPU_data, size_t size){
	size_t nErrors = 0;
	
	for(size_t i = 0; i<size; i++){
		for(size_t j = 0; j<size; j++){
			size_t pos = i*size + j;
			if(CPU_data[pos]!=GPU_data[pos]) nErrors++;
		}
	}
	
	return(nErrors);
}

//------------------------------------------------<

//------------------------------------------------>
//---------- Data imports

long int File_size_row_signal(ifstream &FILEIN){
	std::size_t count=0;
	FILEIN.seekg(0,ios::beg);
	for(std::string line; std::getline(FILEIN, line); ++count){}
	return((long int)count);
}

int Load_kernel(char *filename, int *kernel_size, int *kernel_stride, int *kernel_oversampling, double *constant, double **data){
	double value;
	int file_size, cislo, error;
	error = 0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		error=0;
		file_size = File_size_row_signal(FILEIN) - 1;

		if(file_size>0){
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			FILEIN >> value;
			(*kernel_size) = (int) value;
			FILEIN >> value;
			(*kernel_stride) = (int) value;
			FILEIN >> value;
			(*kernel_oversampling) = (int) value;
			FILEIN >> value;
			(*constant) = value;
			
			printf("File size:%d;\n", file_size );
			printf("Kernel size: %d;\n", (*kernel_size));
			printf("Kernel stride: %d;\n", (*kernel_stride));
			printf("Kernel oversampling: %d;\n", (*kernel_oversampling));
			printf("Constant: %e;\n", (*constant));
			int expected_size = (*kernel_stride)*(*kernel_oversampling);
			printf("Expected size: %d;\n", expected_size);

			*data = (double *)malloc(expected_size*sizeof(double));
			memset( (*data), 0.0, expected_size*sizeof(double));
			if(*data==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
			
			cislo=0;
			while (!FILEIN.eof() && error==0){
				FILEIN >> value;
				(*data)[cislo] = value;
				cislo++;
			}
			
			printf("Final size: %d last value %e;\n", cislo, value);
		}
		else {
			printf("\nFile is void of any content!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}


int Load_grid(char *filename, int *grid_z, int *grid_y, int *grid_x, double2 **data){
	double real, imag, value;
	int file_size, cislo, error;
	error = 0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		error=0;
		file_size = File_size_row_signal(FILEIN) - 1;

		if(file_size>0){
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			FILEIN >> value;
			(*grid_z) = (int) value;
			FILEIN >> value;
			(*grid_y) = (int) value;
			FILEIN >> value;
			(*grid_x) = (int) value;
			
			printf("File size:%d;\n", file_size );
			printf("Grid z: %d;\n", (*grid_z));
			printf("Grid y: %d;\n", (*grid_y));
			printf("Grid x: %d;\n", (*grid_x));
			int expected_size = (*grid_z)*(*grid_y)*(*grid_x);
			printf("Expected size: %d;\n", expected_size);

			*data = (double2 *)malloc(expected_size*sizeof(double2));
			memset( (*data), 0.0, expected_size*sizeof(double2));
			if(*data==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
			
			cislo=0;
			while (!FILEIN.eof() && error==0){
				FILEIN >> real >> imag;
				(*data)[cislo].x = real;
				(*data)[cislo].y = imag;
				cislo++;
			}
			
			printf("Final size: %d last value [%e; %e];\n", cislo, real, imag);
		}
		else {
			printf("\nFile is void of any content!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}


int Load_visibilities(char *filename, int *nVisibilities, double **u_pos, double **v_pos, double **w_pos){
	double u, v, w, value;
	int file_size, cislo, error;
	error = 0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		error=0;
		file_size = File_size_row_signal(FILEIN) - 1;

		if(file_size>0){
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			FILEIN >> value;
			(*nVisibilities) = (int) value;
			FILEIN >> value;
			
			printf("File size:%d;\n", file_size );
			printf("Number visibilities: %d;\n", (*nVisibilities));
			int expected_size = (*nVisibilities);
			printf("Expected size: %d;\n", expected_size);

			*u_pos = (double *)malloc(expected_size*sizeof(double));
			*v_pos = (double *)malloc(expected_size*sizeof(double));
			*w_pos = (double *)malloc(expected_size*sizeof(double));
			memset( (*u_pos), 0.0, expected_size*sizeof(double));
			memset( (*v_pos), 0.0, expected_size*sizeof(double));
			memset( (*w_pos), 0.0, expected_size*sizeof(double));
			if(*u_pos==NULL || *v_pos==NULL || *w_pos==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
			
			cislo=0;
			while (!FILEIN.eof() && error==0){
				FILEIN >> u >> v >> w;
				(*u_pos)[cislo] = u;
				(*v_pos)[cislo] = v;
				(*w_pos)[cislo] = w;
				cislo++;
			}
			
			printf("Final size: %d last value [%e; %e; %e];\n", cislo, u, v, w);
		}
		else {
			printf("\nFile is void of any content!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}

//------------------------------------------------<



//-----------------------------------------------
//---------- Data exports
void Export(int *data, size_t size, const char *filename){
	std::ofstream FILEOUT;
	FILEOUT.open(filename);
	for(size_t i = 0; i<size; i++){
		FILEOUT << data[i] << " ";
	}
	FILEOUT.close();
}
//------------------------------------------------<


int Generate_random(float *h_input, float *base, size_t size, int nSubgrids, int randomize){
	for(int s = 0; s < nSubgrids; s++){
		for(size_t i = 0; i<size; i++){
			h_input[s*size + i] = base[i];
			if(randomize) h_input[s*size + i] = h_input[s*size + i] + (rand() / (float)RAND_MAX)/100.0;
		}
	}
	return(0);
}

int Generate_random(double *h_input, double *base, size_t size, int nSubgrids, int randomize){
	for(int s = 0; s < nSubgrids; s++){
		for(size_t i = 0; i<size; i++){
			h_input[s*size + i] = base[i];
			if(randomize) h_input[s*size + i] = h_input[s*size + i] + (rand() / (float)RAND_MAX)/0.01;
		}
	}
	return(0);
}

int Generate_random(double2 *h_input, double2 *base, size_t size, int nSubgrids, int randomize){
	for(int s = 0; s < nSubgrids; s++){
		for(size_t i = 0; i<size; i++){
			h_input[s*size + i] = base[i];
			if(randomize) {
				h_input[s*size + i].x = h_input[s*size + i].x + (rand() / (float)RAND_MAX)/100.0;
				h_input[s*size + i].y = h_input[s*size + i].y + (rand() / (float)RAND_MAX)/100.0;
			}
		}
	}
	return(0);
}

void calculate_coordinates(
		int grid_size, //dimension of the image's subgrid grid_size x grid_size x 4?
		int x_stride, // padding in x dimension
		int y_stride, // padding in y dimension
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
	
    *grid_offset = (home_y-kernel_size/2)*y_stride + (home_x-kernel_size/2)*x_stride;
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
}


void custom_degrid(
		struct double2 *vis_values,
		struct double2 *grid,
		int grid_z,
		int grid_y,
		int grid_x,
		double *kernel_data,
		int c_kernel_stride,
		int c_kernel_oversampling,
		double *wkernel_data,
		int c_wkernel_stride,
		int c_wkernel_oversampling,
		double *u_vis_coordinates,
		double *v_vis_coordinates,
		double *w_vis_coordinates,
		int c_vis_count,
		double theta,
		double wstep,
		bool conjugate
	) {
	int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
	for(int v = 0; v < c_vis_count; v++){
		
		calculate_coordinates(
			grid_x, 1, grid_y,
			c_kernel_stride, c_kernel_stride, c_kernel_oversampling,
			c_wkernel_stride, c_wkernel_stride, c_wkernel_oversampling,
			theta, wstep, 
			u_vis_coordinates[v], 
			v_vis_coordinates[v], 
			w_vis_coordinates[v],
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);
		
		double vis_r = 0, vis_i = 0;
		for (int z = 0; z < c_wkernel_stride; z++) {
			double visz_r = 0, visz_i = 0;
			for (int y = 0; y < c_kernel_stride; y++) {
				double visy_r = 0, visy_i = 0;
				for (int x = 0; x < c_kernel_stride; x++) {
					double grid_r = 0; //
					double grid_i = 0; //
					struct double2 temp = grid[z*grid_x*grid_y + grid_offset + y*grid_y + x];
					grid_r = temp.x;
					grid_i = temp.y;
					//printf("grid_r = %e; grid_i = %e;\n", grid_r, grid_i);
					visy_r += kernel_data[sub_offset_x + x] * grid_r;
					visy_i += kernel_data[sub_offset_x + x] * grid_i;
				}
				visz_r += kernel_data[sub_offset_y + y] * visy_r;
				visz_i += kernel_data[sub_offset_y + y] * visy_i;
			}
			vis_r += wkernel_data[sub_offset_z + z] * visz_r;
			vis_i += wkernel_data[sub_offset_z + z] * visz_i;
		}
		vis_values[v].x = vis_r;
		if(conjugate) vis_values[v].y = -vis_i;
		else vis_values[v].y = vis_i;
	}
	
}


// GPU function definitions
int GPU_SKA_degrid(
		double2 *output_visibilities, 
		double *gcf_uv_kernel, 
		int uv_kernel_size, 
		int uv_kernel_stride, 
		int uv_kernel_oversampling, 
		double *gcf_w_kernel, 
		int w_kernel_size, 
		int w_kernel_stride, 
		int w_kernel_oversampling, 
		double2 *subgrid, 
		int grid_z, 
		int grid_y, 
		int grid_x, 
		double *u_vis_pos, 
		double *v_vis_pos, 
		double *w_vis_pos, 
		int nVisibilities, 
		int nSubgrids, 
		double theta,
		double wstep,
		int nRuns,
		int device, 
		double *execution_time
	);



int main(int argc, char* argv[]) {
	int nSubgrids = 1;
	int device = 0;
	int nRuns = 1;
	
	//--------------> User input
	char * pEnd;
	if (argc!=2) {
		printf("Argument error!\n");
		printf("1) nSubgrids\n");
        return (1);
    }
	if (argc==2) {
		nSubgrids = strtol(argv[1],&pEnd,10);
	}
	
	if(DEBUG){
		printf("Program arguments:\n");
		printf("nSubgrids: %d;\n", nSubgrids);
	}
	
	double *base_gcf_uv_kernel;
	double *base_gcf_w_kernel;
	double2 *base_subgrid;
	double *base_u_vis_pos;
	double *base_v_vis_pos;
	double *base_w_vis_pos;
	
	double *gcf_uv_kernel;
	int uv_kernel_size = 8;
	int uv_kernel_stride = 8;
	int uv_kernel_oversampling = 16384;
	double *gcf_w_kernel;
	int w_kernel_size = 4;
	int w_kernel_stride = 4;
	int w_kernel_oversampling = 16384;
	double2 *subgrid;
	int grid_z = 4;
	int grid_y = 512;
	int grid_x = 512;
	double *u_vis_pos;
	double *v_vis_pos;
	double *w_vis_pos;
	int nVisibilities = 64;
	double2 *output_visibilities;
	double2 *CPU_output_visibilities;
	
	double theta = 0.999;
	double wstep = 0.999;
	
	char str[200];
	printf("---------- Reading uv kernel -----------\n");
	sprintf(str, "uv_kernel_0.dat");
	Load_kernel(str, &uv_kernel_size, &uv_kernel_stride, &uv_kernel_oversampling, &theta, &base_gcf_uv_kernel);
	printf("---------- Reading w kernel -----------\n");
	sprintf(str, "w_kernel_0.dat");
	Load_kernel(str, &w_kernel_size, &w_kernel_stride, &w_kernel_oversampling, &wstep, &base_gcf_w_kernel);
	printf("------- Reading subgrid values --------\n");
	sprintf(str, "subgrid_0.dat");
	Load_grid(str, &grid_z, &grid_y, &grid_x, &base_subgrid);
	printf("---- Reading visibility positions -----\n");
	sprintf(str, "visibility_position_0.dat");
	Load_visibilities(str, &nVisibilities, &base_u_vis_pos, &base_v_vis_pos, &base_w_vis_pos);
	
	//----------------> Results
	Performance_results SKA_degrid_results;
	SKA_degrid_results.Assign(
			uv_kernel_size, 
			uv_kernel_stride, 
			uv_kernel_oversampling, 
			w_kernel_size, 
			w_kernel_stride, 
			w_kernel_oversampling, 
			grid_z, 
			grid_y, 
			grid_x, 
			nVisibilities, 
			nSubgrids,
			nRuns, 
			"SKA_degrid_results.txt", 
			"mk1"
		);
	double execution_time = 0;
	
	if (DEBUG) printf("\t\tWelcome\n");

	
	size_t uv_kernel_size_in_bytes             = uv_kernel_stride*uv_kernel_oversampling*sizeof(double);
	size_t w_kernel_size_in_bytes              = w_kernel_stride*w_kernel_oversampling*sizeof(double);
	size_t subgrid_size_in_bytes               = grid_z*grid_y*grid_x*sizeof(double2)*nSubgrids;
	size_t visibilities_position_size_in_bytes = nVisibilities*sizeof(double)*nSubgrids;
	size_t output_size_in_bytes                = nVisibilities*sizeof(double2)*nSubgrids;
	size_t total_memory_required_in_bytes      = uv_kernel_size_in_bytes + w_kernel_size_in_bytes + subgrid_size_in_bytes + visibilities_position_size_in_bytes + output_size_in_bytes;

	if (VERBOSE) printf("Input + output size: %f MB;\n", (total_memory_required_in_bytes)/(1024.0*1024.0));
	
	if (VERBOSE) printf("\nHost memory allocation...\t");
		gcf_uv_kernel = (double *)malloc(uv_kernel_size_in_bytes);
		if(gcf_uv_kernel==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		gcf_w_kernel = (double *)malloc(w_kernel_size_in_bytes);
		if(gcf_w_kernel==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		subgrid = (double2 *)malloc(subgrid_size_in_bytes);
		if(subgrid==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		u_vis_pos = (double *)malloc(visibilities_position_size_in_bytes);
		if(u_vis_pos==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		v_vis_pos = (double *)malloc(visibilities_position_size_in_bytes);
		if(v_vis_pos==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		w_vis_pos = (double *)malloc(visibilities_position_size_in_bytes);
		if(w_vis_pos==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		output_visibilities = (double2 *)malloc(output_size_in_bytes);
		if(output_visibilities==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
		CPU_output_visibilities = (double2 *)malloc(output_size_in_bytes);
		if(CPU_output_visibilities==NULL) {printf("\nError in allocation of the memory!\n"); exit(1);}
	if (VERBOSE) printf("done.");

	if (VERBOSE) printf("\nHost memory memset...\t\t");
		memset(gcf_uv_kernel, 0, uv_kernel_size_in_bytes);
		memset(gcf_w_kernel, 0, w_kernel_size_in_bytes);
		memset(subgrid, 0, subgrid_size_in_bytes);
		memset(u_vis_pos, 0, visibilities_position_size_in_bytes);
		memset(v_vis_pos, 0, visibilities_position_size_in_bytes);
		memset(w_vis_pos, 0, visibilities_position_size_in_bytes);
		memset(output_visibilities, 0, output_size_in_bytes);
	if (VERBOSE) printf("done.");

	if (VERBOSE) printf("\nRandom data set...\t\t");
		fflush(stdout);
		srand(time(NULL));
		Generate_random(gcf_uv_kernel, base_gcf_uv_kernel, uv_kernel_stride*uv_kernel_oversampling, 1, 0);
		Generate_random(gcf_w_kernel, base_gcf_w_kernel, w_kernel_stride*w_kernel_oversampling, 1, 0);
		Generate_random(subgrid, base_subgrid, grid_z*grid_y*grid_x, nSubgrids, 1);
		Generate_random(u_vis_pos, base_u_vis_pos, nVisibilities, nSubgrids, 1);
		Generate_random(v_vis_pos, base_v_vis_pos, nVisibilities, nSubgrids, 1);
		Generate_random(w_vis_pos, base_w_vis_pos, nVisibilities, nSubgrids, 1);
	if (VERBOSE) printf("done.\n");
	
	
	//--------------------- SKA degrid ------------
	printf("------------ Distance matrix --------------\n");
	GPU_SKA_degrid(
			output_visibilities, 
			gcf_uv_kernel, 
			uv_kernel_size, 
			uv_kernel_stride, 
			uv_kernel_oversampling, 
			gcf_w_kernel, 
			w_kernel_size, 
			w_kernel_stride, 
			w_kernel_oversampling, 
			subgrid, 
			grid_z, 
			grid_y, 
			grid_x, 
			u_vis_pos, 
			v_vis_pos, 
			w_vis_pos, 
			nVisibilities, 
			nSubgrids, 
			theta,
			wstep,
			nRuns, 
			device, 
			&execution_time
	);
	SKA_degrid_results.degrid_time = execution_time;
	if(VERBOSE) printf("    SKA degrid execution time:\033[32m%0.3f\033[0mms\n", SKA_degrid_results.degrid_time);

	if (CHECK){
		double terror = 0, merror = 0;
		for(int s=0; s<nSubgrids; s++){
			custom_degrid(
				&CPU_output_visibilities[s*nVisibilities],
				&subgrid[s*grid_z*grid_y*grid_x],
				grid_z, grid_y, grid_x,
				gcf_uv_kernel, uv_kernel_stride, uv_kernel_oversampling,
				gcf_w_kernel, w_kernel_stride, w_kernel_oversampling,
				&u_vis_pos[s*nVisibilities], &v_vis_pos[s*nVisibilities], &w_vis_pos[s*nVisibilities],
				nVisibilities,
				theta, wstep,
				false
			);
			for(int f = 0; f < nVisibilities; f++){
				double2 CPU, GPU, diff;
				CPU = CPU_output_visibilities[s*nVisibilities + f];
				GPU = output_visibilities[s*nVisibilities + f];
				diff.x = CPU.x - GPU.x;
				diff.y = CPU.y - GPU.y;
				terror = terror + (diff.x + diff.y)/2.0;
				//printf("Visibility %d = [%e; %e]; GPU: [%e; %e]; Difference: [%e; %e]\n", f, CPU.x, CPU.y, GPU.x, GPU.y, diff.x, diff.y);
			}
		}
		merror = terror / ((double)(nVisibilities*nSubgrids));
		printf("Total error: %e; Mean error: %e;\n", terror, merror);
			
	}
	printf("\n\n");
	//----------------------------------------------<

	
	free(base_gcf_uv_kernel);
	free(base_gcf_w_kernel);
	free(base_subgrid);
	free(base_u_vis_pos);
	free(base_v_vis_pos);
	free(base_w_vis_pos);
	
	free(gcf_uv_kernel);
	free(gcf_w_kernel);
	free(subgrid);
	free(u_vis_pos);
	free(v_vis_pos);
	free(w_vis_pos);
	free(output_visibilities);
	free(CPU_output_visibilities);

	return (0);
}
