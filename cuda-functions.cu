#include "cuda-functions.h"

#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>

//helpers
static int calc_drid_dim(int array_size, int block_size)
{
	return (array_size - 1) / block_size + 1;
}

__global__ void kernel_find_distances_cycle_y_horiz(cv::cuda::PtrStep<float> ptr_dist, cv::cuda::PtrStep<uint8_t> ptr_mat, cv::cuda::PtrStep<uint8_t> ptr_mask, int shift, int ybeg, int size, int xbeg, int xend, int l_straight)
{
	size_t y = blockIdx.x * blockDim.x + threadIdx.x + ybeg;
	
	if (y >= ybeg + size)
		return;

	auto pdist = ptr_dist.ptr(y);
	auto pmat = ptr_mat.ptr(y);
	auto pmask = ptr_mask.ptr(y);

	int x = xbeg;
	while (x != xend)
	{
		if (pmask[x])
		{
			x += shift;
			continue;
		}
		if (pdist[x - shift] + l_straight < pdist[x])
		{
			pdist[x] = pdist[x - shift] + l_straight;
			pmat[x] = pmat[x - shift];
		}
		x += shift;
	}
}

__global__ void kernel_find_distances_cycle_x(float *pdist, uint8_t *pnums, const uint8_t *pmask, const float *pdist_prev, const uint8_t *pnums_prev, int l_straight, int l_diag, int size)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= size || pmask[x] || pdist[x] == 0)
		return;

	if (pdist_prev[x] + l_straight < pdist[x])
	{
		pdist[x] = pdist_prev[x] + l_straight;
		pnums[x] = pnums_prev[x];
	}

	if (x != 0)
	{
		if (pdist_prev[x - 1] + l_diag < pdist[x])
		{
			pdist[x] = pdist_prev[x - 1] + l_diag;
			pnums[x] = pnums_prev[x - 1];
		}
	}

	if (x != (size - 1))
	{
		if (pdist_prev[x + 1] + l_diag < pdist[x])
		{
			pdist[x] = pdist_prev[x + 1] + l_diag;
			pnums[x] = pnums_prev[x + 1];
		}
	}
}

__global__ void kernel_init_seamdist(cv::cuda::PtrStepSz<int> ptr_dist, cv::cuda::PtrStep<uint8_t> ptr_nums, cv::cuda::PtrStep<uint8_t> ptr_outmask, cv::cuda::PtrStep<uint8_t> *ptr_masks, int size_masks)
{
	size_t pt = blockIdx.x * blockDim.x + threadIdx.x;
	if (pt >= ptr_dist.cols * ptr_dist.rows)
		return;
	int y = pt / ptr_dist.cols;
	int x = pt - ptr_dist.cols * y;

	int dist_max = 2 * (ptr_dist.cols + ptr_dist.rows);

	int* pdist = ptr_dist.ptr(y);
	uint8_t* pnums = ptr_nums.ptr(y);
	uint8_t* poutmask = ptr_outmask.ptr(y);

	int count = 0;
	int num = 0;
	for (int i = 0; i < size_masks; ++i)
		if (ptr_masks[i].ptr(y)[x]) //1-visible
		{
			++count;
			num = i;
		}

	if (count == 0)
		poutmask[x] = 1;
	else
		poutmask[x] = 0;

	if (count == 1)
	{
		pdist[x] = 0;
		pnums[x] = num;
	}
	else
	{
		pdist[x] = dist_max;
		pnums[x] = 0;
	}
}

__global__ void kernel_find_seamdistances_cycle_x(const uint8_t **pmasks, int numimages, const uint8_t *poutmask, int *pdist, const int *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev, int l_straight, int l_diag, int size)
{
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size || pdist[x] == 0)
		return;

	if (pdist_prev[x] + l_straight < pdist[x] && (pmasks[pnums_prev[x]][x] || poutmask[x]))
	{
		pdist[x] = pdist_prev[x] + l_straight;
		pnums[x] = pnums_prev[x];
	}

	if (x != 0)
	{
		if (pdist_prev[x - 1] + l_diag < pdist[x] && (pmasks[pnums_prev[x - 1]][x] || poutmask[x]))
		{
			pdist[x] = pdist_prev[x - 1] + l_diag;
			pnums[x] = pnums_prev[x - 1];
		}
	}

	if (x != (size - 1))
	{
		if (pdist_prev[x + 1] + l_diag < pdist[x] && (pmasks[pnums_prev[x + 1]][x] || poutmask[x]))
		{
			pdist[x] = pdist_prev[x + 1] + l_diag;
			pnums[x] = pnums_prev[x + 1];
		}
	}

}

__global__ void kernel_find_seamdistances_cycle_y_horiz(cv::cuda::PtrStep<int> ptr_dist, cv::cuda::PtrStep<uint8_t> ptr_mat, cv::cuda::PtrStep<uint8_t> ptr_outmask, cv::cuda::PtrStep<uint8_t> *ptr_masks, int shift, int ybeg, int size, int xbeg, int xend, int l_straight)
{
	size_t y = blockIdx.x * blockDim.x + threadIdx.x + ybeg;
	if (y >= ybeg + size)
		return;

	auto pdist = ptr_dist.ptr(y);
	auto pmat = ptr_mat.ptr(y);
	auto poutmask = ptr_outmask.ptr(y);
		
	int x = xbeg;
	while (x != xend)
	{
		if (pdist[x] == 0)
		{
			x += shift;
			continue;
		}
		if (pdist[x - shift] + l_straight < pdist[x] && (ptr_masks[pmat[x - shift]].ptr(y)[x] || poutmask[x]))
		{
			pdist[x] = pdist[x - shift] + l_straight;
			pmat[x] = pmat[x - shift];
		}
		x += shift;
	}
}

__global__ void kernel_extract_masks(cv::cuda::PtrStepSz<uint8_t> *ptr_masks, cv::cuda::PtrStepSz<uint8_t> ptr_cvseams, int mask_value)
{
	size_t pt = blockIdx.x * blockDim.x + threadIdx.x;
	if (pt >= ptr_cvseams.cols * ptr_cvseams.rows)
		return;
	int y = pt / ptr_cvseams.cols;
	int x = pt - ptr_cvseams.cols * y;

	auto pseam = ptr_cvseams.ptr(y);
	ptr_masks[pseam[x]].ptr(y)[x] = mask_value;
}

void cuda_find_distances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight)
{
	int size = yend - ybeg;
	if (size < 1)
		return;

	int nthreads = 256;
	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(calc_drid_dim(size, block_dim.x * block_dim.y), 1);

	cv::cuda::PtrStep<float> ptr_dist = dist;
	cv::cuda::PtrStep<uint8_t> ptr_mat = mat;
	cv::cuda::PtrStep<uint8_t> ptr_mask = mask;

	/*cudaEvent_t start, kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&kernel);
	cudaEventRecord(start, 0);*/
	kernel_find_distances_cycle_y_horiz <<<grid_dim, block_dim >>>(ptr_dist, ptr_mat, ptr_mask, shift, ybeg, size, xbeg, xend, l_straight);
	/*cudaEventRecord(kernel, 0);
	cudaEventSynchronize(kernel);

	float time_kernel;
	cudaEventElapsedTime(&time_kernel, start, kernel);
	printf("cuda time_kernel: %f ms\n", time_kernel);*/
}

void cuda_find_distances_cycle_x(
	const uint8_t *pmask, float *pdist, const float *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag)
{
	int size = tmp_xend - tmp_xbeg;
	if (size < 1)
		return;

	int nthreads = 256;
	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(calc_drid_dim(size, block_dim.x * block_dim.y), 1);
	
	float *pdist_beg = pdist + tmp_xbeg;
	uint8_t *pnums_beg = pnums + tmp_xbeg;
	const uint8_t *pmask_beg = pmask + tmp_xbeg;
	const float *pdist_prev_beg = pdist_prev + tmp_xbeg;
	const uint8_t *pnums_prev_beg = pnums_prev + tmp_xbeg;

	/*cudaEvent_t start, kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&kernel);
	cudaEventRecord(start, 0);*/
	kernel_find_distances_cycle_x<<<grid_dim, block_dim>>>(pdist_beg, pnums_beg, pmask_beg, pdist_prev_beg, pnums_prev_beg, l_straight, l_diag, size);
	/*cudaEventRecord(kernel, 0);
	cudaEventSynchronize(kernel);

	float time_kernel;
	cudaEventElapsedTime(&time_kernel, start, kernel);
	printf("cuda time_kernel: %f ms\n", time_kernel);*/
}

void cuda_init_seamdist(cv::cuda::GpuMat &dist, cv::cuda::GpuMat &nums, cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks)
{
	int size = dist.cols * dist.rows;
	if (size < 1)
		return;

	int nthreads = 256;
	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(calc_drid_dim(size, block_dim.x * block_dim.y), 1);

	cv::cuda::PtrStepSz<int> ptr_dist = dist;
	cv::cuda::PtrStep<uint8_t> ptr_nums = nums;
	cv::cuda::PtrStep<uint8_t> ptr_outmask = outmask;
	int numimages = masks.size();
	std::vector<cv::cuda::PtrStep<uint8_t> > ptr_masks(numimages);
	for (int i = 0; i < numimages; ++i)
		ptr_masks[i] = masks[i];

	cv::cuda::PtrStep<uint8_t> *dev_ptr_masks;
	int bufsize = numimages * sizeof(cv::cuda::PtrStep<uint8_t>);
	cudaMalloc(&dev_ptr_masks, bufsize);
	cudaMemcpy(dev_ptr_masks, &ptr_masks[0], bufsize, cudaMemcpyHostToDevice);

	/*cudaEvent_t start, kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&kernel);
	cudaEventRecord(start, 0);*/
	kernel_init_seamdist<<<grid_dim, block_dim>>>(ptr_dist, ptr_nums, ptr_outmask, dev_ptr_masks, numimages);
	/*cudaEventRecord(kernel, 0);
	cudaEventSynchronize(kernel);

	float time_kernel;
	cudaEventElapsedTime(&time_kernel, start, kernel);
	printf("cuda time_kernel: %f ms\n", time_kernel);*/
	cudaFree(dev_ptr_masks);
}

void cuda_find_seamdistances_cycle_x(
	const std::vector<const uint8_t*> &pmasks, const uint8_t *poutmask, int *pdist, const int *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag)
{
	int size = tmp_xend - tmp_xbeg;
	if (size < 1)
		return;

	int nthreads = 256;
	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(calc_drid_dim(size, block_dim.x * block_dim.y), 1);

	int numimages = pmasks.size();
	std::vector<const uint8_t *> pmasks_beg(numimages);
	for (int i = 0; i < numimages; ++i)
		pmasks_beg[i] = pmasks[i] + tmp_xbeg;
	const uint8_t **dev_pmasks_beg;
	int bufsize = numimages * sizeof(const uint8_t *);
	cudaMalloc(&dev_pmasks_beg, bufsize);
	cudaMemcpy(dev_pmasks_beg, &pmasks[0], bufsize, cudaMemcpyHostToDevice);

	const uint8_t *poutmask_beg = poutmask + tmp_xbeg;
	int *pdist_beg = pdist + tmp_xbeg;
	const int *pdist_prev_beg = pdist_prev + tmp_xbeg;
	uint8_t *pnums_beg = pnums + tmp_xbeg;
	const uint8_t *pnums_prev_beg = pnums_prev + tmp_xbeg;

	/*cudaEvent_t start, kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&kernel);
	cudaEventRecord(start, 0);*/
	kernel_find_seamdistances_cycle_x<<<grid_dim, block_dim>>>(dev_pmasks_beg, numimages, poutmask_beg, pdist_beg, pdist_prev_beg, pnums_beg, pnums_prev_beg, l_straight, l_diag, size);
	/*cudaEventRecord(kernel, 0);
	cudaEventSynchronize(kernel);

	float time_kernel;
	cudaEventElapsedTime(&time_kernel, start, kernel);
	printf("cuda time_kernel: %f ms\n", time_kernel);*/
	cudaFree(dev_pmasks_beg);
}

void cuda_find_seamdistances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight)
{
	int size = yend - ybeg;
	if (size < 1)
		return;

	int nthreads = 256;
	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(calc_drid_dim(size, block_dim.x * block_dim.y), 1);

	cv::cuda::PtrStep<int> ptr_dist = dist;
	cv::cuda::PtrStep<uint8_t> ptr_mat = mat;
	const cv::cuda::PtrStep<uint8_t> ptr_outmask = outmask;

	int numimages = masks.size();
	std::vector<cv::cuda::PtrStep<uint8_t> > ptr_masks(numimages);
	for (int i = 0; i < numimages; ++i)
		ptr_masks[i] = masks[i];
	cv::cuda::PtrStep<uint8_t> *dev_ptr_masks;
	int bufsize = numimages * sizeof(cv::cuda::PtrStep<uint8_t>);
	cudaMalloc(&dev_ptr_masks, bufsize);
	cudaMemcpy(dev_ptr_masks, &ptr_masks[0], bufsize, cudaMemcpyHostToDevice);

	/*cudaEvent_t start, kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&kernel);
	cudaEventRecord(start, 0);*/
	kernel_find_seamdistances_cycle_y_horiz<<<grid_dim, block_dim>>>(ptr_dist, ptr_mat, ptr_outmask, dev_ptr_masks, shift, ybeg, size, xbeg, xend, l_straight);
	/*cudaEventRecord(kernel, 0);
	cudaEventSynchronize(kernel);

	float time_kernel;
	cudaEventElapsedTime(&time_kernel, start, kernel);
	printf("cuda time_kernel: %f ms\n", time_kernel);*/
	cudaFree(dev_ptr_masks);
}

void cuda_extract_masks(std::vector<std::vector<cv::cuda::GpuMat> > &cvmaskpyramids, const cv::cuda::GpuMat &cvseams, int mask_value)
{
	int size = cvseams.cols * cvseams.rows;
	if (size < 1)
		return;

	int nthreads = 256;
	dim3 block_dim(nthreads, 1);
	dim3 grid_dim(calc_drid_dim(size, block_dim.x * block_dim.y), 1);

	cv::cuda::PtrStepSz<uint8_t> ptr_cvseams = cvseams;

	int numimages = cvmaskpyramids.size();
	std::vector<cv::cuda::PtrStepSz<uint8_t> > ptr_masks(numimages);
	for (int i = 0; i < numimages; ++i)
		ptr_masks[i] = cvmaskpyramids[i][0];
	cv::cuda::PtrStepSz<uint8_t> *dev_ptr_masks;
	int bufsize = numimages * sizeof(cv::cuda::PtrStepSz<uint8_t>);
	cudaMalloc(&dev_ptr_masks, bufsize);
	cudaMemcpy(dev_ptr_masks, &ptr_masks[0], bufsize, cudaMemcpyHostToDevice);

	/*cudaEvent_t start, kernel;
	cudaEventCreate(&start);
	cudaEventCreate(&kernel);
	cudaEventRecord(start, 0);*/
	kernel_extract_masks<<<grid_dim, block_dim>>>(dev_ptr_masks, ptr_cvseams, mask_value);
	/*cudaEventRecord(kernel, 0);
	cudaEventSynchronize(kernel);

	float time_kernel;
	cudaEventElapsedTime(&time_kernel, start, kernel);
	printf("cuda time_kernel: %f ms\n", time_kernel);*/
	cudaFree(dev_ptr_masks);
}

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