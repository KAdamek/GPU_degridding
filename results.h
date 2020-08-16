#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

class Performance_results{
public:
	char filename[200];
	char kernel[10];
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
	double degrid_time;
	int nRuns;
	
	Performance_results() {
		degrid_time = 0;
	}
	
	void Save(){
		ofstream FILEOUT;
		FILEOUT.open (filename, std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << uv_kernel_size << " " << uv_kernel_stride << " " << uv_kernel_oversampling << " " << w_kernel_size << " " << w_kernel_stride << " " << w_kernel_oversampling << " " << grid_z << " " << grid_y << " " << grid_x << " " << nVisibilities << " " << nSubgrids << " " << nRuns << " " << degrid_time << " " << kernel << endl;
		FILEOUT.close();
	}
	
	void Print(){
		cout << std::fixed << std::setprecision(8) << uv_kernel_size << " " << uv_kernel_stride << " " << uv_kernel_oversampling << " " << w_kernel_size << " " << w_kernel_stride << " " << w_kernel_oversampling << " " << grid_z << " " << grid_y << " " << grid_x << " " << nVisibilities << " " << nSubgrids << " " << nRuns << " " << degrid_time << " " << kernel << endl;
	}
	
	void Assign(
			int t_uv_kernel_size, 
			int t_uv_kernel_stride, 
			int t_uv_kernel_oversampling, 
			int t_w_kernel_size, 
			int t_w_kernel_stride, 
			int t_w_kernel_oversampling, 
			int t_grid_z, 
			int t_grid_y, 
			int t_grid_x, 
			int t_nVisibilities, 
			int t_nSubgrids, 
			int t_nRuns, 
			const char *t_filename, 
			const char *t_kernel
		){
		uv_kernel_size =          t_uv_kernel_size;
		uv_kernel_stride =        t_uv_kernel_stride;
		uv_kernel_oversampling =  t_uv_kernel_oversampling;
		w_kernel_size =           t_w_kernel_size;
		w_kernel_stride =         t_w_kernel_stride;
		w_kernel_oversampling =   t_w_kernel_oversampling;
		grid_z =                  t_grid_z;
		grid_y =                  t_grid_y;
		grid_x =                  t_grid_x;
		nVisibilities =           t_nVisibilities;
		nSubgrids =               t_nSubgrids;
		nRuns        = t_nRuns;

		sprintf(filename,"%s", t_filename);
		sprintf(kernel,"%s",t_kernel);
	}
	
};
