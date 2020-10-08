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

class subgrid_params {
	public:
	std::vector<double> min_u;
	std::vector<double> max_u;
	std::vector<double> min_v;
	std::vector<double> max_v;
	std::vector<double> min_w;
	std::vector<double> max_w;
	
	void add_limits(double t_min_u, double t_max_u, double t_min_v, double t_max_v, double t_min_w, double t_max_w){
		min_u.push_back(t_min_u);
		min_v.push_back(t_min_v);
		min_w.push_back(t_min_w);
		max_u.push_back(t_max_u);
		max_v.push_back(t_max_v);
		max_w.push_back(t_max_w);
	}
};


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


size_t get_subgrid_total_size(int start_file, int end_file, int *grid_z, int *grid_y, int *grid_x, int *nSubgrids){
	size_t total_subgrid_size = 0;
	(*nSubgrids) = 0;
	
	double value;
	int error = 0;
	int file_number = start_file;
	int done = 0;
	char filename[200];
	ifstream FILEIN;
	while(done==0 && file_number < end_file){
		sprintf(filename, "subgrid_%d.dat", file_number);
		FILEIN.open(filename,ios::in);
		if (!FILEIN.fail()){
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			if(file_number==start_file){
				FILEIN >> value;
				(*grid_z) = (int) value;
				FILEIN >> value;
				(*grid_y) = (int) value;
				FILEIN >> value;
				(*grid_x) = (int) value;
			}
			else {
				FILEIN >> value;
				if( (*grid_z) != (int) value ) {
					printf("ERROR! grid_z != value: grid_z = %d; value = %d;\n", (*grid_z), (int) value);
					error++;
				}
				FILEIN >> value;
				if( (*grid_y) != (int) value ) {
					printf("ERROR! grid_y != value: grid_y = %d; value = %d;\n", (*grid_y), (int) value);
					error++;
				}
				FILEIN >> value;
				if( (*grid_x) != (int) value ) {
					printf("ERROR! grid_x != value: grid_x = %d; value = %d;\n", (*grid_x), (int) value);
					error++;
				}
			}
			if(error>0) return(0);
			
			size_t expected_size = (*grid_z)*(*grid_y)*(*grid_x);
			total_subgrid_size = total_subgrid_size + expected_size;
			
			file_number++;
			(*nSubgrids)++;
			
			FILEIN.close();
		}
		else {
			done = 1;
		}
		
	}
	return(total_subgrid_size);
}

int Generate_grid_from_zero(int start_file, int end_file, int *grid_z, int *grid_y, int *grid_x, int *nSubgrids, size_t *subgrid_size, double2 **data){
	char filename[200];
	size_t expected_size, total_subgrid_size;
	double value;
	ifstream FILEIN;
	sprintf(filename, "subgrid_0.dat");
	if(start_file!=end_file) return(1);
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		FILEIN.clear();
		FILEIN.seekg(0,ios::beg);
		
		FILEIN >> value;
		(*grid_z) = (int) value;
		FILEIN >> value;
		(*grid_y) = (int) value;
		FILEIN >> value;
		(*grid_x) = (int) value;
		
		expected_size = (*grid_z)*(*grid_y)*(*grid_x);
		total_subgrid_size = end_file*expected_size;
		FILEIN.close();
	}
	else {
		return(1);
	}
	
	*data = (double2 *)malloc(total_subgrid_size*sizeof(double2));
	memset( (*data), 0.0, total_subgrid_size*sizeof(double2));
	if(*data==NULL){
		printf("\nAllocation error!\n");
		return(1);
	}
	(*nSubgrids) = end_file;
	
	size_t index = 0;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		FILEIN.clear();
		FILEIN.seekg(0,ios::beg);
		FILEIN >> value;
		FILEIN >> value;
		FILEIN >> value;
		
		for(size_t f = 0; f < expected_size; f++){
			double real, imag;
			FILEIN >> real >> imag;
			(*data)[index].x = real;
			(*data)[index].y = imag;
			index++;
		}
		
		FILEIN.close();
		printf("Reading of the file finished!\n");
	}
	else {
		printf("Error while opening a file!\n");
		return(1);
	}
	
	double rnd;
	printf("Generating other subgrids...\n");
	for(int sg=1; sg<end_file; sg++){
		for(size_t f = 0; f < expected_size; f++){
			rnd = ((rand() / (double)RAND_MAX) - 0.5)/10;
			(*data)[index].x = (*data)[f].x + rnd;
			rnd = ((rand() / (double)RAND_MAX) - 0.5)/10;
			(*data)[index].y = (*data)[f].y + rnd;
			index++;
		}
		printf(".");
		fflush(stdout);
	}
	
	printf("  Loading subgrids: Input size: %zu bytes = %f MB;\n", total_subgrid_size*sizeof(double2), (total_subgrid_size*sizeof(double2))/(1024.0*1024.0));
	(*subgrid_size) = total_subgrid_size*sizeof(double2);
	return(0);
}

int Load_grid(int start_file, int end_file, int *grid_z, int *grid_y, int *grid_x, int *nSubgrids, size_t *subgrid_size, double2 **data){	
	size_t total_subgrid_size;
	
	total_subgrid_size = get_subgrid_total_size(start_file, end_file, grid_z, grid_y, grid_x, nSubgrids);
	size_t expected_size = (*grid_z)*(*grid_y)*(*grid_x);
	printf("  Loading subgrids: subgrid size = %zu; total size = %zu;\n", expected_size, total_subgrid_size);
	
	if(total_subgrid_size == 0 || total_subgrid_size < expected_size) {
		printf("No files found!\n");
		return(1);
	}
	
	*data = (double2 *)malloc(total_subgrid_size*sizeof(double2));
	memset( (*data), 0.0, total_subgrid_size*sizeof(double2));
	if(*data==NULL){
		printf("\nAllocation error!\n");
		return(1);
	}
	
	printf("  Loading subgrids: Loading %d subgrids...\n  ", (*nSubgrids));
	double real, imag, value;
	size_t index = 0;
	int file_number = start_file;
	int done = 0;
	char filename[200];
	ifstream FILEIN;
	while(done==0 && file_number < end_file){
		sprintf(filename, "subgrid_%d.dat", file_number);
		FILEIN.open(filename,ios::in);
		if (!FILEIN.fail()){
			file_number++;
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			FILEIN >> value;
			FILEIN >> value;
			FILEIN >> value;
			
			for(size_t f = 0; f < expected_size; f++){
				FILEIN >> real >> imag;
				(*data)[index].x = real;
				(*data)[index].y = imag;
				index++;
			}
			printf(".");
			fflush(stdout);
			FILEIN.close();
		}
		else {
			printf("No more files to load!\n");
			done = 1;
		}
	}
	printf("\n");
	printf("  Loading subgrids: Files processed: %d;\n", (*nSubgrids));
	printf("  Loading subgrids: Input size: %zu bytes = %f MB;\n", total_subgrid_size*sizeof(double2), (total_subgrid_size*sizeof(double2))/(1024.0*1024.0));
	(*subgrid_size) = total_subgrid_size*sizeof(double2);
	return(0);
}


int Generate_visibilities_from_zero(
		int start_file, 
		int end_file, 
		std::vector<int> *nVisibilities, 
		int *max_nVisibilities, 
		std::vector<double> *u_pos, 
		std::vector<double> *v_pos, 
		std::vector<double> *w_pos,
		std::vector<double> *min_u,
		std::vector<double> *max_u,
		std::vector<double> *min_v,
		std::vector<double> *max_v,
		std::vector<double> *min_w,
		std::vector<double> *max_w,
		std::vector<double> *du,
		std::vector<double> *dv,
		std::vector<double> *dw,
		std::vector<double> *start_u,
		std::vector<double> *start_v,
		std::vector<double> *start_w
		){
	double u, v, w, value;
	double old_u = 0, old_v = 0, old_w = 0;
	int running_sum = 0;
	int max = 0;
	char filename[200];
	double t_min_u, t_max_u, t_min_v, t_max_v, t_min_w, t_max_w;
	if(start_file!=end_file) return(1);
	ifstream FILEIN;
	sprintf(filename, "visibility_position_0.dat");
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		FILEIN.clear();
		FILEIN.seekg(0,ios::beg);
			
		FILEIN >> value;
		nVisibilities->push_back(running_sum); // adding starting 0
		int expected_size = (int) value;
		for(int f=0; f<end_file; f++){
			running_sum = running_sum + (int) value;
			nVisibilities->push_back(running_sum);
		}
		nVisibilities->push_back(running_sum);
		
		FILEIN >> value;
		
		FILEIN >> t_min_u >> t_max_u;
		FILEIN >> t_min_v >> t_max_v;
		FILEIN >> t_min_w >> t_max_w;
		
		for(int f=0; f<end_file; f++){
			min_u->push_back(t_min_u);
			min_v->push_back(t_min_v);
			min_w->push_back(t_min_w);
			max_u->push_back(t_max_u);
			max_v->push_back(t_max_v);
			max_w->push_back(t_max_w);
		}
		
		for(int f = 0; f < expected_size; f++){
			FILEIN >> u >> v >> w;
			u_pos->push_back(u);
			v_pos->push_back(v);
			w_pos->push_back(w);
			if(f==1){
				for(int f=0; f<end_file; f++){
					du->push_back(u - old_u);
					dv->push_back(v - old_v);
					dw->push_back(w - old_w);
					start_u->push_back(old_u);
					start_v->push_back(old_v);
					start_w->push_back(old_w);
				}
			}
			old_u = u;
			old_v = v;
			old_w = w;
		}
		
		FILEIN.close();
	}
	else {
		printf("Error while loading a file!\n");
		return(1);
	}
	
	(*max_nVisibilities) = max;
	return(0);
}

int Load_visibilities(
		int start_file, 
		int end_file, 
		std::vector<int> *nVisibilities, 
		int *max_nVisibilities, 
		std::vector<double> *u_pos, 
		std::vector<double> *v_pos, 
		std::vector<double> *w_pos,
		std::vector<double> *min_u,
		std::vector<double> *max_u,
		std::vector<double> *min_v,
		std::vector<double> *max_v,
		std::vector<double> *min_w,
		std::vector<double> *max_w,
		std::vector<double> *du,
		std::vector<double> *dv,
		std::vector<double> *dw,
		std::vector<double> *start_u,
		std::vector<double> *start_v,
		std::vector<double> *start_w
		){
	double u, v, w, value;
	double old_u, old_v, old_w;
	int file_number = start_file;
	int running_sum = 0;
	int done = 0;
	int max = 0;
	char filename[200];
	double t_min_u, t_max_u, t_min_v, t_max_v, t_min_w, t_max_w;
	ifstream FILEIN;
	while(done==0 && file_number < end_file){
		sprintf(filename, "visibility_position_%d.dat", file_number);
		FILEIN.open(filename,ios::in);
		if (!FILEIN.fail()){
			file_number++;

			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
				
			FILEIN >> value;
			nVisibilities->push_back(running_sum);
			if(value > max) max = value;
			int expected_size = (int) value;
			running_sum = running_sum + (int) value;
			//printf("nVisibilities=%d; running_sum=%d;\n", (int) value, running_sum);
			
			FILEIN >> value;
			
			FILEIN >> t_min_u >> t_max_u;
			FILEIN >> t_min_v >> t_max_v;
			FILEIN >> t_min_w >> t_max_w;
			
			//printf("min a max u: %e - %e\n", t_min_u, t_max_u);
			//printf("min a max u: %e - %e\n", t_min_v, t_max_v);
			//printf("min a max u: %e - %e\n", t_min_w, t_max_w);
			
			min_u->push_back(t_min_u);
			min_v->push_back(t_min_v);
			min_w->push_back(t_min_w);
			max_u->push_back(t_max_u);
			max_v->push_back(t_max_v);
			max_w->push_back(t_max_w);
			
			for(int f = 0; f < expected_size; f++){
				FILEIN >> u >> v >> w;
				u_pos->push_back(u);
				v_pos->push_back(v);
				w_pos->push_back(w);
				if(f==1){
					du->push_back(u - old_u);
					dv->push_back(v - old_v);
					dw->push_back(w - old_w);
					start_u->push_back(old_u);
					start_v->push_back(old_v);
					start_w->push_back(old_w);
				}
				old_u = u;
				old_v = v;
				old_w = w;
			}
			
			FILEIN.close();
		}
		else {
			printf("No more files to load!\n");
			done = 1;
		}
	}
	nVisibilities->push_back(running_sum);
	(*max_nVisibilities) = max;
	return(0);
}

void Generate_visibilities(
		int visibilities_per_subgrid_random,
		int nSubgrids,
		std::vector<int> *nVisibilities, 
		int *max_nVisibilities, 
		std::vector<double> *u_pos, 
		std::vector<double> *v_pos, 
		std::vector<double> *w_pos,
		std::vector<double> *min_u,
		std::vector<double> *max_u,
		std::vector<double> *min_v,
		std::vector<double> *max_v,
		std::vector<double> *min_w,
		std::vector<double> *max_w
	){
	int running_sum = 0;
	*max_nVisibilities = visibilities_per_subgrid_random;
	nVisibilities->push_back(0);
	for(int f = 0; f < nSubgrids; f++){
		double l_min_u = min_u->operator[](f);
		double l_max_u = max_u->operator[](f);
		double l_min_v = min_v->operator[](f);
		double l_max_v = max_v->operator[](f);
		double l_min_w = min_w->operator[](f);
		double l_max_w = max_w->operator[](f);
		double range_u = l_max_u - l_min_u;
		double range_v = l_max_v - l_min_v;
		double range_w = l_max_w - l_min_w;
		double rnd = 0;
		for(int v = 0; v<visibilities_per_subgrid_random; v++){
			rnd = rand() / (float)RAND_MAX;
			u_pos->push_back(rnd*range_u + l_min_u);
			rnd = rand() / (float)RAND_MAX;
			v_pos->push_back(rnd*range_v + l_min_v);
			rnd = rand() / (float)RAND_MAX;
			w_pos->push_back(rnd*range_w + l_min_w);
		}
		running_sum = running_sum + visibilities_per_subgrid_random;
		nVisibilities->push_back(running_sum);
	}
}

void Generate_visibilities_on_line(
		int visibilities_per_line,
		int nSubgrids,
		double scale,
		int test_type,
		std::vector<int> *nVisibilities, 
		int *max_nVisibilities, 
		std::vector<double> *u_pos, 
		std::vector<double> *v_pos, 
		std::vector<double> *w_pos,
		std::vector<double> *min_u,
		std::vector<double> *max_u,
		std::vector<double> *min_v,
		std::vector<double> *max_v,
		std::vector<double> *min_w,
		std::vector<double> *max_w,
		std::vector<double> *du,
		std::vector<double> *dv,
		std::vector<double> *dw,
		std::vector<double> *start_u,
		std::vector<double> *start_v,
		std::vector<double> *start_w
	){
	int running_sum = 0;
	if(test_type==1 || test_type==2){ // fixed amount of visibilities per subgrid
		*max_nVisibilities = visibilities_per_line;
	}
	nVisibilities->push_back(0);
	for(int f = 0; f < nSubgrids; f++){
		int l_nVisibilities = visibilities_per_line;
		if(test_type==2){
			l_nVisibilities = f*(visibilities_per_line/nSubgrids);
		}
		double l_min_u = min_u->operator[](f);
		double l_max_u = max_u->operator[](f);
		double l_min_v = min_v->operator[](f);
		double l_max_v = max_v->operator[](f);
		double l_min_w = min_w->operator[](f);
		double l_max_w = max_w->operator[](f);
		double l_range_u = l_max_u - l_min_u;
		double l_range_v = l_max_v - l_min_v;
		double l_range_w = l_max_w - l_min_w;
		double l_du = du->operator[](f);
		double l_dv = dv->operator[](f);
		double l_dw = dw->operator[](f);
		double l_start_u = start_u->operator[](f);
		double l_start_v = start_v->operator[](f);
		double l_start_w = start_w->operator[](f);
		if(test_type==1 || test_type==2){
			l_start_u = l_min_u;
			l_start_v = l_min_v;
			l_start_w = l_min_w;
			l_du = l_range_u/l_nVisibilities;
			l_dv = l_range_v/l_nVisibilities;
			l_dw = l_range_w/l_nVisibilities;
		}

		int count = 0;
		for(int vis = 0; vis<l_nVisibilities; vis++){
			double u = l_start_u + ((double) vis)*l_du*scale;
			double v = l_start_v + ((double) vis)*l_dv*scale;
			double w = l_start_w + ((double) vis)*l_dw*scale;
			if( (u > l_min_u) && (u < l_max_u) && (v > l_min_v) && (v < l_max_v) && (w > l_min_w) && (w < l_max_w)){
				u_pos->push_back(u);
				v_pos->push_back(v);
				w_pos->push_back(w);
				count++;
			}
		}
		running_sum = running_sum + count;
		nVisibilities->push_back(running_sum);
		if(DEBUG){
			printf("[min; max]\n");
			printf("u: [%e; %e];\n", l_min_u, l_max_u);
			printf("v: [%e; %e];\n", l_min_v, l_max_v);
			printf("w: [%e; %e];\n", l_min_w, l_max_w);
			printf("delta: [%e; %e; %e];\n", l_du, l_dv, l_dw);
			printf("start: [%e; %e; %e];\n", l_start_u, l_start_v, l_start_w);
			printf("Visibilities added: %d;\n", count);
		}
	}
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


void calculate_grid_offset_components(
		int grid_size, //dimension of the image's subgrid grid_size x grid_size x 4?
		int kernel_size, // gcf kernel support
		int oversample, // oversampling of the uv kernel
		double theta, //conversion parameter from uv coordinates to xy coordinates x=u*theta
		double u, // 
		double v, // coordinates of the visibility 
		int *grid_offset_x, // offset in the image subgrid
		int *grid_offset_y // offset in the image subgrid
	){
	// x coordinate
	double x = theta*u;
	double ox = x*oversample;
	//int iox = lrint(ox);
	int iox = round(ox); // round to nearest
    iox += (grid_size / 2 + 1) * oversample - 1;
    int home_x = iox / oversample;
	
	// y coordinate
	double y = theta*v;
	double oy = y*oversample;
	//int iox = lrint(ox);
	int ioy = round(oy);
    ioy += (grid_size / 2 + 1) * oversample - 1;
    int home_y = ioy / oversample;
	
	*grid_offset_y = (home_y-kernel_size/2);
	*grid_offset_x = (home_x-kernel_size/2);
}


void Compute_coordinates_and_bin(
	int grid_y,
	int grid_x,
	int c_kernel_stride,
	int c_kernel_oversampling,
	int c_wkernel_stride,
	int c_wkernel_oversampling,
	double *u_vis_coordinates,
	double *v_vis_coordinates,
	double *w_vis_coordinates,
	std::vector<int> *nVisibilities,
	int nSubgrids,
	double theta,
	double wstep
){
	// Get total number of visibilities
	size_t total_nVisibilities = 0;
	for(int s=0; s<nSubgrids; s++){
		size_t local_nVisibilities = nVisibilities->operator[](s + 1) - nVisibilities->operator[](s);
		total_nVisibilities = total_nVisibilities + local_nVisibilities;
	}
	if(DEBUG) printf("Total visibilities: %zu;\n", total_nVisibilities);
		
	// Allocate array for all the grid_offset
	int *grid_offset_data; 
	grid_offset_data = new int[total_nVisibilities];
	int *bins_per_subgrid;
	bins_per_subgrid = new int[nSubgrids];
	for(int s=0; s<nSubgrids; s++) bins_per_subgrid[s] = 0;
	std::vector<int> nVisibilities_per_bin;
	
	int min_vis_per_bin = total_nVisibilities;
	int max_vis_per_bin = 0;
	int average_vis_per_bin = 0;
	int total_bins = 0;
	
	// populate grid_offset_data
	size_t count = 0;
	int old_grid_offset = 0;
	size_t current_bin = 0;
	int max_nBins = 0;
	for(int s=0; s<nSubgrids; s++){
		size_t local_nVisibilities = nVisibilities->operator[](s + 1) - nVisibilities->operator[](s);
		for(size_t f=0; f<local_nVisibilities; f++){
			int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;
			int grid_offset_x, grid_offset_y;
			calculate_coordinates(
				grid_x, 1, grid_y,
				c_kernel_stride, c_kernel_stride, c_kernel_oversampling,
				c_wkernel_stride, c_wkernel_stride, c_wkernel_oversampling,
				theta, wstep, 
				u_vis_coordinates[count], 
				v_vis_coordinates[count], 
				w_vis_coordinates[count],
				&grid_offset, 
				&sub_offset_x, &sub_offset_y, &sub_offset_z
			);
			
			calculate_grid_offset_components(
				grid_x,
				c_kernel_stride,
				c_kernel_oversampling, 
				theta,
				u_vis_coordinates[count], 
				v_vis_coordinates[count], 
				&grid_offset_x,
				&grid_offset_y
			);
			if(DEBUG) printf("grid_offset: %d; grid_offset_x: %d; grid_offset_y: %d;\n", grid_offset, grid_offset_x, grid_offset_y);
			
			if (f == 0) {
				// starting a new subgrid thus add one to the total number of bins for this subgrid and set old_grid_offset
				bins_per_subgrid[s]++;
				old_grid_offset = grid_offset_x;
				nVisibilities_per_bin.push_back(1);
				current_bin = nVisibilities_per_bin.size() - 1;
			}
			else {
				if(old_grid_offset != grid_offset_x || nVisibilities_per_bin[current_bin] >= 10){
					bins_per_subgrid[s]++;
					if(nVisibilities_per_bin[current_bin]<min_vis_per_bin) min_vis_per_bin = nVisibilities_per_bin[current_bin];
					if(nVisibilities_per_bin[current_bin]>max_vis_per_bin) max_vis_per_bin = nVisibilities_per_bin[current_bin];
					average_vis_per_bin += nVisibilities_per_bin[current_bin];
					total_bins++;
					nVisibilities_per_bin.push_back(1);
					current_bin = nVisibilities_per_bin.size() - 1;
					old_grid_offset = grid_offset_x;
				}
				else {
					nVisibilities_per_bin[current_bin]++;
				}
			}
			grid_offset_data[count] = grid_offset;
			count++;
		}
		if(DEBUG) {
			printf("Number of bin for this subgrid: %d;\n", bins_per_subgrid[s]);
			printf("Number of visibilities per bin: ");
			for(int i=0; i<bins_per_subgrid[s]; i++) printf("%d ", nVisibilities_per_bin[current_bin - i]);
			printf("\n");
			printf("----- NEXT -----\n");
		}
		if ( max_nBins < bins_per_subgrid[s] ) max_nBins = bins_per_subgrid[s];
	}

	average_vis_per_bin = average_vis_per_bin/total_bins;
	printf("Average number of Visibilities per bin: %d;\n", average_vis_per_bin);
	printf("Minimum number of Visibilities per bin: %d;\n", min_vis_per_bin);
	printf("Maximum number of Visibilities per bin: %d;\n", max_vis_per_bin);
	printf("Maximum number of bins per subgrid is %d;\n", max_nBins);
	delete[] grid_offset_data;
	delete[] bins_per_subgrid;
}



// -------------------------> GPU function definitions
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
		size_t total_nVisibilities,
		int *nVisibilities, 
		int max_nVisibilities, 
		int nSubgrids, 
		double theta,
		double wstep,
		int kernel_type,
		int nRuns,
		int device, 
		double *execution_time
	);


void Generate_visibilities_and_run_GPU(
		double *base_gcf_uv_kernel, 
		int uv_kernel_size, 
		int uv_kernel_stride, 
		int uv_kernel_oversampling, 
		double *base_gcf_w_kernel, 
		int w_kernel_size, 
		int w_kernel_stride, 
		int w_kernel_oversampling, 
		double2 *base_subgrid, 
		int grid_z, 
		int grid_y, 
		int grid_x, 
		int nSubgrids, 
		double theta,
		double wstep,
		int start_file,
		int end_file,
		int visibilities_per_subgrid_random,
		int visibilities_per_line,
		double step_multiple,
		int kernel_type,
		int test_type,
		int nRuns,
		int device
) {
	printf("---- Reading visibility positions -----\n");
	std::vector<double> base_u_vis_pos;
	std::vector<double> base_v_vis_pos;
	std::vector<double> base_w_vis_pos;
	std::vector<double> min_u;
	std::vector<double> max_u;
	std::vector<double> min_v;
	std::vector<double> max_v;
	std::vector<double> min_w;
	std::vector<double> max_w;
	std::vector<int> nVisibilities;
	
	std::vector<double> du;
	std::vector<double> dv;
	std::vector<double> dw;
	std::vector<double> start_u;
	std::vector<double> start_v;
	std::vector<double> start_w;
	
	int max_nVisibilities;
	if(start_file==end_file){
		Generate_visibilities_from_zero(
			start_file, 
			end_file, 
			&nVisibilities, 
			&max_nVisibilities, 
			&base_u_vis_pos, 
			&base_v_vis_pos, 
			&base_w_vis_pos,
			&min_u,
			&min_v,
			&min_w,
			&max_u,
			&max_v,
			&max_w,
			&du,
			&dv,
			&dw,
			&start_u,
			&start_v,
			&start_w
		);
	}
	else {
		Load_visibilities(
			start_file, 
			end_file, 
			&nVisibilities, 
			&max_nVisibilities, 
			&base_u_vis_pos, 
			&base_v_vis_pos, 
			&base_w_vis_pos,
			&min_u,
			&min_v,
			&min_w,
			&max_u,
			&max_v,
			&max_w,
			&du,
			&dv,
			&dw,
			&start_u,
			&start_v,
			&start_w
		);
	}
		
	if(visibilities_per_subgrid_random>0){
		printf("Generating new visibility positions at random.\n");
		nVisibilities.clear();
		base_u_vis_pos.clear();
		base_v_vis_pos.clear();
		base_w_vis_pos.clear();
		Generate_visibilities(
			visibilities_per_subgrid_random,
			nSubgrids,
			&nVisibilities, 
			&max_nVisibilities, 
			&base_u_vis_pos, 
			&base_v_vis_pos, 
			&base_w_vis_pos,
			&min_u,
			&min_v,
			&min_w,
			&max_u,
			&max_v,
			&max_w
		);
	}
	
	if(visibilities_per_line>0){
		printf("Generating new visibility positions on the line.\n");
		nVisibilities.clear();
		base_u_vis_pos.clear();
		base_v_vis_pos.clear();
		base_w_vis_pos.clear();
		Generate_visibilities_on_line(
			visibilities_per_line,
			nSubgrids,
			step_multiple,
			test_type,
			&nVisibilities, 
			&max_nVisibilities, 
			&base_u_vis_pos, 
			&base_v_vis_pos, 
			&base_w_vis_pos,
			&min_u,
			&min_v,
			&min_w,
			&max_u,
			&max_v,
			&max_w,
			&du,
			&dv,
			&dw,
			&start_u,
			&start_v,
			&start_w
		);
	}
	
	size_t total_nVisibilities                 = base_u_vis_pos.size();
	size_t visibilities_position_size_in_bytes = base_u_vis_pos.size()*sizeof(double);
	size_t output_size_in_bytes                = base_u_vis_pos.size()*sizeof(double2);
	printf("visibilities_position_size_in_bytes = %zu;\n", visibilities_position_size_in_bytes);
	printf("output_size_in_bytes = %zu;\n", output_size_in_bytes);
	printf("max_nVisibilities = %d;\n", max_nVisibilities);
	
	double2 *output_visibilities;
	double2 *CPU_output_visibilities;
	output_visibilities = (double2 *)malloc(output_size_in_bytes);
	if(output_visibilities==NULL) {
		printf("\nError in allocation of the memory!\n"); 
		exit(1);
	}
	CPU_output_visibilities = (double2 *)malloc(output_size_in_bytes);
	if(CPU_output_visibilities==NULL) {
		printf("\nError in allocation of the memory!\n"); 
		exit(1);
	}
	memset(output_visibilities, 0, output_size_in_bytes);
	memset(CPU_output_visibilities, 0, output_size_in_bytes);
	
	//----------------> Results
	Performance_results SKA_degrid_results;
	char kernel_str[5];
	char vis_type_str[200];
	sprintf(kernel_str, "mk%d", kernel_type);
	if(visibilities_per_subgrid_random>0) sprintf(vis_type_str, "random");
	if(visibilities_per_line>0) sprintf(vis_type_str, "line");
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
			total_nVisibilities, 
			nSubgrids,
			nRuns, 
			"SKA_degrid_results.txt", 
			kernel_str,
			vis_type_str,
			test_type
		);
	double execution_time = 0;
	
	//--------------------- SKA degrid ------------
	printf("------------ SKA degrid --------------\n");
	GPU_SKA_degrid(
			output_visibilities, 
			base_gcf_uv_kernel, 
			uv_kernel_size, 
			uv_kernel_stride, 
			uv_kernel_oversampling, 
			base_gcf_w_kernel, 
			w_kernel_size, 
			w_kernel_stride, 
			w_kernel_oversampling, 
			base_subgrid, 
			grid_z, 
			grid_y, 
			grid_x, 
			base_u_vis_pos.data(), 
			base_v_vis_pos.data(), 
			base_w_vis_pos.data(),
			base_u_vis_pos.size(),
			nVisibilities.data(), 
			max_nVisibilities,
			nSubgrids, 
			theta,
			wstep,
			kernel_type,
			nRuns, 
			device, 
			&execution_time
	);
	
	SKA_degrid_results.degrid_time = execution_time;
	if(VERBOSE) printf("    SKA degrid execution time:\033[32m%0.3f\033[0mms\n", SKA_degrid_results.degrid_time);
	if(visibilities_per_subgrid_random>0) SKA_degrid_results.nVisibilities = visibilities_per_subgrid_random;
	if(visibilities_per_line>0) SKA_degrid_results.nVisibilities = visibilities_per_line;
	SKA_degrid_results.Save();
	
	if (CHECK){
		double terror = 0, merror = 0;
		for(int s=0; s<nSubgrids; s++){
			int active_nVisibilities = nVisibilities[s+1] - nVisibilities[s];
			custom_degrid(
				&CPU_output_visibilities[nVisibilities[s]],
				&base_subgrid[s*grid_z*grid_y*grid_x],
				grid_z, grid_y, grid_x,
				base_gcf_uv_kernel, uv_kernel_stride, uv_kernel_oversampling,
				base_gcf_w_kernel, w_kernel_stride, w_kernel_oversampling,
				&base_u_vis_pos[nVisibilities[s]], &base_v_vis_pos[nVisibilities[s]], &base_w_vis_pos[nVisibilities[s]],
				active_nVisibilities,
				theta, wstep,
				false
			);
			for(int f = 0; f < active_nVisibilities; f++){
				double2 CPU, GPU, diff;
				CPU = CPU_output_visibilities[nVisibilities[s] + f];
				GPU = output_visibilities[nVisibilities[s] + f];
				diff.x = CPU.x - GPU.x;
				diff.y = CPU.y - GPU.y;
				terror = terror + (diff.x + diff.y)/2.0;
				//printf("Visibility %d = [%e; %e]; GPU: [%e; %e]; Difference: [%e; %e]\n", f, CPU.x, CPU.y, GPU.x, GPU.y, diff.x, diff.y);
			}
		}
		merror = terror / ((double)(total_nVisibilities));
		printf("Total error: %e; Mean error: %e;\n", terror, merror);
			
	}
	printf("\n\n");
	
	/*
	Compute_coordinates_and_bin(
		grid_y, 
		grid_x,
		uv_kernel_stride, 
		uv_kernel_oversampling,
		w_kernel_stride, 
		w_kernel_oversampling,
		base_u_vis_pos.data(), 
		base_v_vis_pos.data(), 
		base_w_vis_pos.data(),
		&nVisibilities,
		nSubgrids,
		theta, wstep
	);
	*/
	
	if(output_visibilities != NULL) free(output_visibilities);
	if(CPU_output_visibilities != NULL) free(CPU_output_visibilities);
	
	nVisibilities.clear();
	base_u_vis_pos.clear();
	base_v_vis_pos.clear();
	base_w_vis_pos.clear();
	
	min_u.clear();
	max_u.clear();
	min_v.clear();
	max_v.clear();
	min_w.clear();
	max_w.clear();
	nVisibilities.clear();
	
	du.clear();
	dv.clear();
	dw.clear();
	start_u.clear();
	start_v.clear();
	start_w.clear();
}


int main(int argc, char* argv[]) {
	int start_file = 0;
	int end_file = 1000;
	int visibilities_per_subgrid_random = 0;
	float step_multiple = 1.0;
	int visibilities_per_line = 0;
	int device = 0;
	int nRuns = 1;
	
	//--------------> User input
	char * pEnd;
	if (argc!=2) {
		printf("Argument error!\n");
		printf("1) Number of subgrids\n");
        return (1);
    }
	if (argc==2) {
		end_file                 = strtol(argv[1],&pEnd,10);
		start_file               = end_file;
	}
	
	/*
	if(DEBUG){
		printf("Program arguments:\n");
		printf("start_file: %d;\n", start_file);
		printf("end_file: %d;\n", end_file);
		printf("visibilities_per_subgrid_random: %d;\n", visibilities_per_subgrid_random);
		printf("step in the visibilities: %e;\n", step_multiple);
		printf("visibilities_per_line: %d;\n", visibilities_per_line);
	}
	*/
	
	
	if(start_file>end_file) {
		printf("start_file<=end_file!\n");
		return(1);
	}
	
	
	char str[200];
	printf("---------- Reading uv kernel -----------\n");
	sprintf(str, "uv_kernel_0.dat");
	double *base_gcf_uv_kernel = NULL;
	int uv_kernel_size, uv_kernel_stride, uv_kernel_oversampling;
	double theta;
	Load_kernel(str, &uv_kernel_size, &uv_kernel_stride, &uv_kernel_oversampling, &theta, &base_gcf_uv_kernel);

	
	printf("---------- Reading w kernel -----------\n");
	sprintf(str, "w_kernel_0.dat");
	double *base_gcf_w_kernel = NULL;
	int w_kernel_size, w_kernel_stride, w_kernel_oversampling;
	double wstep;
	Load_kernel(str, &w_kernel_size, &w_kernel_stride, &w_kernel_oversampling, &wstep, &base_gcf_w_kernel);
	
	printf("------- Reading subgrid values --------\n");
	size_t total_subgrid_size;
	double2 *base_subgrid = NULL;
	int grid_z, grid_y, grid_x, nSubgrids;
	if(start_file==end_file) Generate_grid_from_zero(start_file, end_file, &grid_z, &grid_y, &grid_x, &nSubgrids, &total_subgrid_size, &base_subgrid);
	else Load_grid(start_file, end_file, &grid_z, &grid_y, &grid_x, &nSubgrids, &total_subgrid_size, &base_subgrid);
	printf("Total subgrid size: %zu bytes = %f MB;\n", total_subgrid_size, total_subgrid_size/(1024.0*1024.0));
	printf("nSubgrids=%d;\n", nSubgrids);

	
	int test_type;
	
	std::vector<int> nVis_test = {50, 100, 500, 1000, 10000, 100000, 1000000};
	
	//--------------- random positions
	test_type = 1;
	for(int f=0; f<(int)nVis_test.size(); f++){
		for(int kernel_type=1; kernel_type<=6; kernel_type++){
			Generate_visibilities_and_run_GPU(
				base_gcf_uv_kernel, uv_kernel_size, uv_kernel_stride, uv_kernel_oversampling, 
				base_gcf_w_kernel, w_kernel_size, w_kernel_stride, w_kernel_oversampling, 
				base_subgrid, grid_z, grid_y, grid_x, nSubgrids, theta,	wstep,
				start_file,
				end_file,
				nVis_test[f], //random
				0, // line
				step_multiple,
				kernel_type,
				test_type,
				nRuns, device
				);
		}
	}
	
	//--------------- line
	test_type = 1;
	for(int f=0; f<(int)nVis_test.size(); f++){
		for(int kernel_type=1; kernel_type<=6; kernel_type++){
			Generate_visibilities_and_run_GPU(
				base_gcf_uv_kernel, uv_kernel_size, uv_kernel_stride, uv_kernel_oversampling, 
				base_gcf_w_kernel, w_kernel_size, w_kernel_stride, w_kernel_oversampling, 
				base_subgrid, grid_z, grid_y, grid_x, nSubgrids, theta,	wstep,
				start_file,
				end_file,
				0, //random
				nVis_test[f], // line
				step_multiple,
				kernel_type,
				test_type,
				nRuns, device
				);
		}
	}

	test_type = 2;
	for(int f=0; f<(int)nVis_test.size(); f++){
		for(int kernel_type=1; kernel_type<=6; kernel_type++){
			Generate_visibilities_and_run_GPU(
				base_gcf_uv_kernel, uv_kernel_size, uv_kernel_stride, uv_kernel_oversampling, 
				base_gcf_w_kernel, w_kernel_size, w_kernel_stride, w_kernel_oversampling, 
				base_subgrid, grid_z, grid_y, grid_x, nSubgrids, theta,	wstep,
				start_file,
				end_file,
				0, //random
				nVis_test[f], // line
				step_multiple,
				kernel_type,
				test_type,
				nRuns, device
				);
		}
	}
	
	if(base_gcf_uv_kernel != NULL) free(base_gcf_uv_kernel);
	if(base_gcf_w_kernel != NULL) free(base_gcf_w_kernel);
	if(base_subgrid != NULL) free(base_subgrid);

	return (0);
}
