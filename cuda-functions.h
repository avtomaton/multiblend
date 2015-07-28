#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#ifndef NO_CUDA

#include <cstdint>
#include <opencv2/core/core.hpp>

void cuda_get_memory(size_t *free, size_t *total);

void cuda_find_distances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight);

void cuda_find_distances_cycle_x(
	const uint8_t *pmask, float *pdist, const float *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag);

void cuda_init_seamdist(cv::cuda::GpuMat &dist, cv::cuda::GpuMat &nums, cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks);

void cuda_find_seamdistances_cycle_x(
	const std::vector<const uint8_t*> &pmasks, const uint8_t *poutmask, int *pdist, const int *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag);

void cuda_find_seamdistances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight);

void cuda_extract_masks(std::vector<std::vector<cv::cuda::GpuMat> > &cvmaskpyramids, const cv::cuda::GpuMat &cvseams, int mask_value);

void cuda_mask_into_output(std::vector<cv::cuda::GpuMat> &outs, const std::vector<cv::cuda::GpuMat> &ins, const cv::cuda::GpuMat &mask, const cv::Size &ofs, int max_value);

void cuda_pyrDown(const cv::cuda::GpuMat &umat, cv::cuda::GpuMat &lmat);

void cuda_pyrUp(const cv::cuda::GpuMat &lmat, cv::cuda::GpuMat &umat);

/*
void cuda_init();
void* cuda_stream_create();
void cuda_stream_destroy(void* stream);

diy::Point* cuda_points_alloc(int size);
diy::Point* cuda_host_alloc(int size);

void cuda_points_free(diy::Point *points);
void cuda_host_free(diy::Point *points);

void cuda_points_to_device(const diy::Point *host_data, diy::Point *device_data, int size, void *stream);
void cuda_points_from_device(const diy::Point *device_data, diy::Point *host_data, int size, void *stream);

void cuda_StreamSynchronize(void *stream);

int cuda_simple_test(diy::Point *points, int size, void *stream);
int cuda_init_mat(
	const diy::Point *map, diy::Point *ocv_map,
	size_t rows, size_t cols, int channels, real_t finish_cx, real_t finish_cy);
*/

#endif

#endif //CUDA_FUNCTIONS_H
