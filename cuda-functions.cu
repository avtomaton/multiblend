#include "cuda-functions.h"
#ifndef NO_CUDA
#include <cuda_runtime.h>

/*
__device__ __constant__ int MAXITER = 100;
__device__ __constant__ real_t M_PI_DEG = 180.0f;
__device__ __constant__ real_t M_EPSILON  = 1e-6f; //enough precosion for --use_fast_math
#undef M_PI
__device__ __constant__ real_t M_PI = 3.14159265358979323846;

__device__ __constant__ int cwhere[4];

static const int nthreads = 256;



__global__ void kernel_init_mat(const diy::Point *map, diy::Point *ocv_map, size_t rows, size_t cols, size_t bufsize, real_t finish_cx, real_t finish_cy, size_t offset)
{
	size_t pt_index = blockIdx.x * blockDim.x + threadIdx.x + offset;

	extern __shared__ diy::Point sh_mem_point[];
	diy::Point *shvar = &sh_mem_point[0];

	while (pt_index < bufsize + offset)
	{
		size_t y = pt_index / cols;
		size_t x = pt_index - y * cols;

		shvar[threadIdx.x] = map[y * cols + x];
		
		shvar[threadIdx.x].x = float(shvar[threadIdx.x].x + finish_cx);
		shvar[threadIdx.x].y = float(shvar[threadIdx.x].y + finish_cy);
		
		ocv_map[pt_index-offset] = shvar[threadIdx.x];
		//ocv_map[pt_index].x = float(invar.x + finish_cx);
		//ocv_map[pt_index].y = float(invar.y + finish_cy);

		pt_index += blockDim.x * gridDim.x;
	}
}



















static inline int func_finalize(void *stream)
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("cuda enter status: %s\n", cudaGetErrorString(error));
		return -1;
	}
	error = cudaStreamSynchronize(*(cudaStream_t*)stream);
	if (error != cudaSuccess)
	{
		printf("cuda exit status: %s\n", cudaGetErrorString(error));
		return -1;
	}
	return 0;
}

void cuda_StreamSynchronize(void *stream)
{
	cudaStreamSynchronize(*(cudaStream_t*)stream);
}


//API
void cuda_init()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d devices\n", deviceCount);
	cudaDeviceSynchronize();
}

void* cuda_stream_create()
{
	cudaStream_t *stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
	if (!stream)
		return 0;
	cudaError_t res = cudaStreamCreate(stream);
	if (res == cudaSuccess)
		return stream;
	else
		return 0;
}

void cuda_stream_destroy(void* stream)
{
	cudaError_t res = cudaStreamDestroy(*((cudaStream_t*)stream));
	free(stream);
	if (res != cudaSuccess)
		; //emit error
}

diy::Point* cuda_points_alloc(int size)
{
	diy::Point *ptr = 0;
	cudaError_t res = cudaMalloc(&ptr, size * sizeof(diy::Point));
	if (res == cudaSuccess)
		return ptr;
	else
		return 0;
}

diy::Point* cuda_host_alloc(int size)
{
	diy::Point *ptr = 0;
	cudaError_t res = cudaHostAlloc((void**)&ptr, size * sizeof(diy::Point), cudaHostAllocDefault);
	if (res == cudaSuccess)
		return ptr;
	else
		return 0;
}

void cuda_points_free(diy::Point *points)
{
	cudaError_t res = cudaFree(points);
	if (res != cudaSuccess)
		; //emit error
}

void cuda_host_free(diy::Point *points)
{
	cudaError_t res = cudaFreeHost(points);
	if (res != cudaSuccess)
		; //emit error
}

void cuda_points_to_device(const diy::Point *host_data, diy::Point *device_data, int size, void *stream)
{
	//TODO: error checking
	cudaMemcpyAsync(device_data, host_data, size * sizeof(diy::Point), cudaMemcpyHostToDevice, *((cudaStream_t*)stream));
}

void cuda_points_from_device(const diy::Point *device_data, diy::Point *host_data, int size, void *stream)
{
	//TODO: error checking
	cudaMemcpyAsync(host_data, device_data, size * sizeof(diy::Point), cudaMemcpyDeviceToHost, *((cudaStream_t*)stream));
}








int cuda_init_mat(
	const diy::Point *map, diy::Point *ocv_map,
	size_t rows, size_t cols, int channels, real_t finish_cx, real_t finish_cy)
{
	
	cudaEvent_t start, props, maloc, kernel, fre;
cudaEventCreate(&start);
cudaEventCreate(&props);
cudaEventCreate(&maloc);
cudaEventCreate(&kernel);
cudaEventCreate(&fre);

cudaEventRecord(start, 0);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int nthreads = 256;
	int nblocks = prop.multiProcessorCount * 128;

	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(nblocks, 1);

	int shared_memory = nthreads * sizeof(diy::Point);
	int size = rows * cols;
	diy::Point *map_dev;
	diy::Point *ocv_map_dev;

cudaEventRecord(props, 0);
cudaEventSynchronize(props);
	size_t N = 2;
	size_t blocksize = (size / N) + 1;
	
	cudaMalloc(&map_dev, blocksize * sizeof(diy::Point));
	cudaMalloc(&ocv_map_dev, blocksize * sizeof(diy::Point));

cudaEventRecord(maloc, 0);
cudaEventSynchronize(maloc);

	printf("size = %d\n", size);
	for (size_t block = 0; block < N; ++block)
	{
		size_t bufsize = (block == N - 1) ? (size - block*blocksize) : blocksize;
		size_t offset = block * blocksize;
		printf("bufsize[%d] = %d\n", block, bufsize);
		printf("offset[%d] = %d\n", block, offset);

		cudaMemcpy(map_dev, &map[offset], bufsize * sizeof(diy::Point), cudaMemcpyHostToDevice);

		kernel_init_mat<<<nblocks, block_dim, shared_memory>>>(map_dev, ocv_map_dev, rows, cols, bufsize, finish_cx, finish_cy, offset);

		cudaMemcpy(&ocv_map[offset], ocv_map_dev, bufsize * sizeof(diy::Point), cudaMemcpyDeviceToHost);
	
		printf("ok\n");
	}


cudaEventRecord(kernel, 0);
cudaEventSynchronize(kernel);

	cudaFree(map_dev);
	cudaFree(ocv_map_dev);

cudaEventRecord(fre, 0);
cudaEventSynchronize(fre);

float time_props, time_maloc, time_kernel, time_fre, time_all;
cudaEventElapsedTime(&time_props, start, props);
cudaEventElapsedTime(&time_maloc, props, maloc);
cudaEventElapsedTime(&time_kernel, maloc, kernel);
cudaEventElapsedTime(&time_fre, kernel, fre);
cudaEventElapsedTime(&time_all, start, fre);

printf("cuda time_props: %f ms\n", time_props);
printf("cuda time_maloc: %f ms\n", time_maloc);
printf("cuda time_kernel: %f ms\n", time_kernel);
printf("cuda time_fre: %f ms\n", time_fre);
printf("cuda time_all(%f): %f ms\n\n", (rows*cols*sizeof(diy::Point)) / (1024.0*1024.0), time_all);

	return 0;
}
*/
#endif
