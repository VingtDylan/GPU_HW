#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "util.h"

using namespace std;
using namespace cv;

#pragma once
#ifdef __INTELLISENSE__
	void __syncthreads();
#endif

__host__ void displayInfo();
__global__ void GPU_minChannel_Kernel(uchar*, uchar*, int);
__global__ void GPU_darkChannel_Kernel(uchar*, uchar*, int, int);
__global__ void GPU_transmission_Kernel(uchar*, float*, uchar*, int, float, float);
__global__ void GPU_recover_Kernel(uchar*, float*, uchar*, uchar*, int);
__host__ Mat GPU_minChannel(Mat);
__host__ Mat GPU_darkChannel(Mat);
__host__ Mat GPU_transmission(Mat, Vec3b, float, float);
__host__ Mat GPU_recover(Mat, Mat, Vec3b);
__host__ void errCatch(cudaError_t);

int main() {
	displayInfo();
	Mat tohaze = imread("extra1.jpg");
	Mat minChannel = GPU_minChannel(tohaze);
	Mat darkChannel = GPU_darkChannel(minChannel);
	Vec3b A = atmosphic(tohaze, darkChannel);
	float w = 0.85f, t0 = 0.1f;
	Mat tx = GPU_transmission(tohaze, A, w, t0);
	Mat recover = GPU_recover(tohaze, tx, A);

	imshow("hazed image", tohaze);
	imshow("Dehazed image", recover);
	waitKey(0);
	return 0;
}

__global__
void GPU_minChannel_Kernel(uchar* _tohaze_, uchar* _minChannel_, int col)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int _row = tx / col;
	int _col = tx % col;
	uchar t_inf = 0xff;
	for (int t = 0; t < 3; t++) {
		t_inf = t_inf > _tohaze_[3 * tx + t] ? _tohaze_[3 * tx + t] : t_inf;
	}
	_minChannel_[_row * col + _col] = t_inf;
}

__global__
void GPU_darkChannel_Kernel(uchar* _minChannel_, uchar* _darkChannel_, int row, int col)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int _row = tx / col;
	int _col = tx % col;

	int m = 7, n = 7;

	int xl, xr, yl, yr;
	xl = _row - m + 1 > 0 ? _row - m + 1 : 0; 
	xr = _row + m - 1 < row ? _row + m - 1 : row;
	yl = _col - n + 1 > 0 ? _col - n + 1 : 0;
	yr = _col + n - 1 < col ? _col + n - 1 : col;

	uchar t_inf = 0xff;
	for (int k = xl; k < xr; ++k)
	{
		for (int l = yl; l < yr; ++l)
		{
			uchar s = _minChannel_[k * col + l];
			t_inf = t_inf > s ? s : t_inf;
		}
	}
	_darkChannel_[_row * col + _col] = t_inf;
}

__global__ 
void GPU_transmission_Kernel(uchar* _tohaze_, float* _tx_, uchar* _a_, int col, float w, float t0)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int _row = tx / col;
	int _col = tx % col;

	_tx_[_row * col + _col] = (float)_tohaze_[3 * tx + 0] / _a_[0];
	_tx_[_row * col + _col] = _tx_[_row * col + _col] > (float)_tohaze_[3 * tx + 1] / _a_[1] ? (float)_tohaze_[3 * tx + 1] / _a_[1] : _tx_[_row * col + _col];
	_tx_[_row * col + _col] = _tx_[_row * col + _col] > (float)_tohaze_[3 * tx + 2] / _a_[2] ? (float)_tohaze_[3 * tx + 2] / _a_[2] : _tx_[_row * col + _col];

	_tx_[_row * col + _col] = 1.0 - w * _tx_[_row * col + _col] > t0 ? 1.0 - w * _tx_[_row * col + _col] : t0;
}

__global__
void GPU_recover_Kernel(uchar* _tohaze_, float* _tx_, uchar* _a_, uchar* _res_, int col)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int _row = tx / col;
	int _col = tx % col;

	for (int i = 0; i < 3; i++)
	{
		int t = _tohaze_[3 * (_row * col + _col) + i] - _a_[i];
		int s = int(t / _tx_[_row * col + _col] + _a_[i]);
		//printf("%d %d %d %d %d\n", int(_tohaze_[3 * (_row * col + _col) + i]),int(_a_[i]),int(t), int(t / _tx_[_row * col + _col]), s);
		s = s > 0xff ? 0xff : s;
		_res_[3 * (_row * col + _col) + i] = uchar(s);
		//printf(" %d \n", int(_res_[3 * (_row * col + _col) + i]));
	}
}

__host__ 
void displayInfo() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	//cout << deviceCount << endl;
	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		cout << "使用GPU device " << i << ": " << devProp.name << endl;
		cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << endl;
		cout << "SM的数量：" << devProp.multiProcessorCount << endl;
		cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
		cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << endl;
		cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << endl;
		cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << endl;
		cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << endl;
		cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << endl;
		cout << "======================================================" << endl;
	}
}

__host__
Mat GPU_minChannel(Mat img)
{
	CV_Assert(!img.empty());

	int col = img.cols, row = img.rows;
	uchar *_tohaze, *_minChannel;
	uchar *_cuda_tohaze, *_cuda_minChannel;

	_tohaze = (uchar*)malloc(sizeof(uchar) * col * row * 3);
	_minChannel = (uchar*)malloc(sizeof(uchar) * col * row);
	_tohaze = img.data;

	errCatch(cudaMalloc((void**)&_cuda_tohaze, 3 * col * row * sizeof(uchar)));
	errCatch(cudaMalloc((void**)&_cuda_minChannel, col * row * sizeof(uchar)));
	errCatch(cudaMemcpy(_cuda_tohaze, _tohaze, 3 * col * row * sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 dimGrid(row);
	dim3 dimBlock(col);

	//开始计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	GPU_minChannel_Kernel << < dimGrid, dimBlock >> > (_cuda_tohaze, _cuda_minChannel, col);
	cudaThreadSynchronize();
	//停止计时
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elaspedTime;
	cudaEventElapsedTime(&elaspedTime, start, stop);
	cout << "minChannel time: " << elaspedTime << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	errCatch(cudaMemcpy(_minChannel, _cuda_minChannel, col * row * sizeof(uchar), cudaMemcpyDeviceToHost));
	Mat minChannel(row, col, CV_8UC1, _minChannel);
	errCatch(cudaFree(_cuda_tohaze));
	errCatch(cudaFree(_cuda_minChannel));
	return minChannel;
}

__host__
Mat GPU_darkChannel(Mat img)
{
	CV_Assert(!img.empty());

	int col = img.cols, row = img.rows;
	uchar *_minChannel, *_darkChannel;
	uchar *_cuda_minChannel, *_cuda_darkChannel;

	_minChannel = (uchar*)malloc(sizeof(uchar) * col * row);
	_darkChannel = (uchar*)malloc(sizeof(uchar) * col * row);
	_minChannel = img.data;

	errCatch(cudaMalloc((void**)&_cuda_minChannel, col * row * sizeof(uchar)));
	errCatch(cudaMalloc((void**)&_cuda_darkChannel, col * row * sizeof(uchar)));
	errCatch(cudaMemcpy(_cuda_minChannel, _minChannel, col * row * sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 dimGrid(row);
	dim3 dimBlock(col);

	//开始计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	GPU_darkChannel_Kernel << < dimGrid, dimBlock >> > (_cuda_minChannel, _cuda_darkChannel, row, col);
	cudaThreadSynchronize();
	//停止计时
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elaspedTime;
	cudaEventElapsedTime(&elaspedTime, start, stop);
	cout << "darkChannel time: " << elaspedTime << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	errCatch(cudaMemcpy(_darkChannel, _cuda_darkChannel, col * row * sizeof(uchar), cudaMemcpyDeviceToHost));
	Mat darkChannel(row, col, CV_8UC1, _darkChannel);
	errCatch(cudaFree(_cuda_minChannel));
	errCatch(cudaFree(_cuda_darkChannel));
	return darkChannel;
}

__host__
Mat GPU_transmission(Mat img, Vec3b a, float w, float t0)
{
	CV_Assert(!img.empty());

	int col = img.cols, row = img.rows;
	uchar *_tohaze, *_cuda_tohaze;
	float *_tx, *_cuda_tx;
	uchar *_a, *_cuda_a;

	_tohaze = (uchar*)malloc(sizeof(uchar) * col * row * 3);
	_tx = (float*)malloc(sizeof(float) * col * row);
	_a = (uchar *)malloc(sizeof(uchar) * 3);
	_tohaze = img.data;
	for (int i = 0; i < 3; i++) _a[i] = a[i];

	errCatch(cudaMalloc((void**)&_cuda_tohaze, 3 * col * row * sizeof(uchar)));
	errCatch(cudaMalloc((void**)&_cuda_tx, col * row * sizeof(float)));
	errCatch(cudaMalloc((void**)&_cuda_a, 3 * sizeof(uchar)));
	errCatch(cudaMemcpy(_cuda_tohaze, _tohaze, 3 * col * row * sizeof(uchar), cudaMemcpyHostToDevice));
	errCatch(cudaMemcpy(_cuda_a, _a, 3 * sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 dimGrid(row);
	dim3 dimBlock(col);

	//开始计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	GPU_transmission_Kernel << < dimGrid, dimBlock >> > (_cuda_tohaze, _cuda_tx, _cuda_a, col, w, t0);
	cudaThreadSynchronize();
	//停止计时
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elaspedTime;
	cudaEventElapsedTime(&elaspedTime, start, stop);
	cout << "transmission time: " << elaspedTime << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	errCatch(cudaMemcpy(_tx, _cuda_tx, col * row * sizeof(float), cudaMemcpyDeviceToHost));
	Mat tx(row, col, CV_32F, _tx);
	errCatch(cudaFree(_cuda_tohaze));
	errCatch(cudaFree(_cuda_tx));
	errCatch(cudaFree(_cuda_a));
	return tx;
}

__host__
Mat GPU_recover(Mat img, Mat t, Vec3b a)
{
	CV_Assert(!img.empty());

	int col = img.cols, row = img.rows;
	uchar *_tohaze, *_cuda_tohaze;
	float *_tx, *_cuda_tx;
	uchar *_a, *_cuda_a;
	uchar *_res, *_cuda_res;

	_tohaze = (uchar*)malloc(sizeof(uchar) * col * row * 3);
	_tx = (float*)malloc(sizeof(float) * col * row);
	_a = (uchar *)malloc(sizeof(uchar) * 3);
	_res = (uchar*)malloc(sizeof(uchar) * col * row * 3);
	_tohaze = img.data;
	for (int i = 0; i < row; i++)for (int j = 0; j < col; j++) _tx[i * col + j] = t.at<float>(i, j);
	for (int i = 0; i < 3; i++) _a[i] = a[i];

	errCatch(cudaMalloc((void**)&_cuda_tohaze, 3 * col * row * sizeof(uchar)));
	errCatch(cudaMalloc((void**)&_cuda_tx, col * row * sizeof(float)));
	errCatch(cudaMalloc((void**)&_cuda_a, 3 * sizeof(uchar)));
	errCatch(cudaMalloc((void**)&_cuda_res, 3 * col * row * sizeof(uchar)));
	errCatch(cudaMemcpy(_cuda_tohaze, _tohaze, 3 * col * row * sizeof(uchar), cudaMemcpyHostToDevice));
	errCatch(cudaMemcpy(_cuda_tx, _tx, col * row * sizeof(float), cudaMemcpyHostToDevice));
	errCatch(cudaMemcpy(_cuda_a, _a, 3 * sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 dimGrid(row);
	dim3 dimBlock(col);

	//开始计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	GPU_recover_Kernel << < dimGrid, dimBlock >> > (_cuda_tohaze, _cuda_tx, _cuda_a, _cuda_res, col);
	cudaThreadSynchronize();
	//停止计时
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elaspedTime;
	cudaEventElapsedTime(&elaspedTime, start, stop);
	cout << "recover time: " << elaspedTime << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	errCatch(cudaMemcpy(_res, _cuda_res, 3 * col * row * sizeof(uchar), cudaMemcpyDeviceToHost));
	Mat res(row, col, CV_8UC3, _res);
	errCatch(cudaFree(_cuda_tohaze));
	errCatch(cudaFree(_cuda_tx));
	errCatch(cudaFree(_cuda_a));
	errCatch(cudaFree(_cuda_res));
	return res;
}

__host__
void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}