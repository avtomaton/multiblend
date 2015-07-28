#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "stdlib.h"
#include "stdio.h"

#include "structs.h"

#include <opencv2/core/core.hpp>


#define L_STRAIGHT 2
#define L_DIAG 3
#define L_STRAIGHT_SEAM 3
#define L_DIAG_SEAM 4
void output(int level, const char* fmt, ...);

void report_time(const char* name, double time);

#ifdef WIN32
#define SNPRINTF _snprintf_s
#else
#define SNPRINTF snprintf
int _stricmp(const char* a, const char* b);
void* _aligned_malloc(size_t size, int boundary);
void _aligned_free(void* a);
void fopen_s(FILE** f, const char* filename, const char* mode);
#endif

void clean_globals();
void clear_temp();
void die(const char* error, ...);

//geotiff
/*void geotiff_register(TIFF* tif);
int geotiff_read(TIFF* tiff, GeoTIFFInfo* info);
*/
#if TIFF_LIBRARY
int geotiff_write(TIFF * tiff, GeoTIFFInfo * info);
#endif

inline float get_l2(const cv::Point &p1, const cv::Point &p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

void print_gpu_memory();

//load images
void trim8(void* bitmap, uint32 w, uint32 h, int bpp, int* top, int* left, int* bottom, int* right);
void trim16(void* bitmap, uint32 w, uint32 h, int bpp, int* top, int* left, int* bottom, int* right);
void trim(void* bitmap, int w, int h, int bpp, int* top, int* left, int* bottom, int* right);
void extract8(struct_image* image, void* bitmap);
void extract_opencv(const cv::Mat &mask, const cv::Mat &channels, struct_image* image, void* bitmap);
void extract16(struct_image* image, void* bitmap);
void extract(struct_image* image, void* bitmap);
void to_cvmat(cv::Mat &mat, struct_image* image);
void inpaint8(struct_image* image, uint32* edt);
void inpaint16(struct_image* image, uint32* edt);
void inpaint(struct_image* image, uint32* edt);
void tighten();

#ifdef NO_CUDA
inline int non_zero_row(const cv::Mat &mask, int y);
inline int non_zero_col(const cv::Mat &mask, int x, int yl, int yr);
inline int non_zero_col(const cv::Mat &mask, int x);
bool is_two_areas(const cv::Mat &mask);
int localize_xl(const cv::Mat &mask, float j0, float jstep, float left, float right);
int localize_xr(const cv::Mat &mask, float j0, float jstep, float left, float right);
int localize_yl(const cv::Mat &mask, float i0, float istep, float left, float right);
int localize_yr(const cv::Mat &mask, float i0, float istep, float left, float right);
void init_dist(const cv::Mat &mask, cv::Mat &dist);
void find_distances_cycle_y_horiz(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight);
inline void find_distances_cycle_x(
	const uint8_t *pmask, float *pdist, const float *pdist_prev, cv::Vec3b *pnums, const cv::Vec3b *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag);
void find_distances_cycle_y_vert(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend, int xl, int xr,
	bool two_areas,
	int l_straight, int l_diag);
int search_l(const cv::Mat &mask, float left, float right, bool isy);
int search_r(const cv::Mat &mask, float left, float right, bool isy);
cv::Rect get_visible_rect(const cv::Mat &mask);
void inpaint_opencv(cv::Mat &mat, const cv::Mat &mask, const cv::Rect &rect);
void mat2struct(int i, const std::string &filename, cv::Mat &matimage, cv::Mat &mask, cv::Mat &dist);
#else
inline int non_zero_row(const cv::cuda::GpuMat &mask, int y);
inline int non_zero_col(const cv::cuda::GpuMat &mask, int x);
bool is_two_areas(const cv::cuda::GpuMat &mask);
int localize_xl(const cv::cuda::GpuMat &mask, float j0, float jstep, float left, float right);
int localize_xr(const cv::cuda::GpuMat &mask, float j0, float jstep, float left, float right);
int localize_yl(const cv::cuda::GpuMat &mask, float i0, float istep, float left, float right);
int localize_yr(const cv::cuda::GpuMat &mask, float i0, float istep, float left, float right);
void init_dist(const cv::cuda::GpuMat &mask, cv::cuda::GpuMat &dist);
void find_distances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight);
inline void find_distances_cycle_x(
	const uint8_t *pmask, float *pdist, const float *pdist_prev, cv::Vec3b *pnums, const cv::Vec3b *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag);
void find_distances_cycle_y_vert(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend, int xl, int xr,
	bool two_areas,
	int l_straight, int l_diag);
int search_l(const cv::cuda::GpuMat &mask, float left, float right, bool isy);
int search_r(const cv::cuda::GpuMat &mask, float left, float right, bool isy);
cv::Rect get_visible_rect(const cv::cuda::GpuMat &mask);
void inpaint_opencv(cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask, const cv::Rect &rect);
void mat2struct(int i, const std::string &filename, std::vector<cv::cuda::GpuMat> &matimages, cv::cuda::GpuMat &mask, cv::cuda::GpuMat &dist);
#endif
void load_images();

//seaming
void seam_png(int mode, const char* filename);
void load_seams();
void rightdownxy();
void leftupxy();
void simple_seam();
void make_seams();
#ifdef NO_CUDA
void find_seamdistances_cycle_y_horiz(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &outmask, const std::vector<cv::Mat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight);
void find_seamdistances_cycle_x(
	const std::vector<const uint8_t*> &pmasks, const uint8_t *poutmask, int *pdist, const int *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag);
void find_seamdistances_cycle_y_vert(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &outmask, const std::vector<cv::Mat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight, int l_diag);
void init_seamdist(cv::Mat &dist, cv::Mat &nums, cv::Mat &outmask, const std::vector<cv::Mat> &masks);
void set_g_edt_opencv(cv::Mat &nums, cv::Mat &outmask, const std::vector<cv::Mat> &masks);
#else
void find_seamdistances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight);
void find_seamdistances_cycle_x(
	const std::vector<const uint8_t*> &pmasks, const uint8_t *poutmask, int *pdist, const int *pdist_prev, uint8_t *pnums, const uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag);
void find_seamdistances_cycle_y_vert(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight, int l_diag);
void init_seamdist(cv::cuda::GpuMat &dist, cv::cuda::GpuMat &nums, cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks);
void set_g_edt_opencv(cv::cuda::GpuMat &nums, cv::cuda::GpuMat &outmask, const std::vector<cv::cuda::GpuMat> &masks);
#endif
void seam();

//maskpyramids
void png_mask(int i);
int squish_line(float* input, float *output, int inwidth, int outwidth);
int squash_lines(float* a, float* b, float* c, float* o, int width);
void shrink_mask(float* input, float **output_pointer, int inwidth, int inheight, int outwidth, int outheight);
void extract_top_masks();
void shrink_masks();
void mask_pyramids();

//blending
void save_out_pyramid(int c, bool collapsed);
void hshrink(struct_level* upper, struct_level* lower);
void vshrink(struct_level* upper, struct_level* lower);
void hps(struct_level* upper, struct_level *lower);
void shrink_hps(struct_level* upper, struct_level* lower);

#ifdef NO_CUDA
void resizedown(const cv::Mat &umat, cv::Mat &lmat, const cv::Size &ofs);
void resizeup(const cv::Mat &lmat, cv::Mat &umat, const cv::Size &ofs = cv::Size(0, 0));
void shrink_opencv(struct_level* upper, struct_level* lower, const cv::Mat &umat, cv::Mat &lmat);
void hps_opencv(struct_level* upper, struct_level* lower, cv::Mat &umat, const cv::Mat &lmat);
void collapse_opencv(const cv::Mat &lower, cv::Mat &upper);
void dither_opencv(cv::Mat &top, cv::Mat &out);
#else
void resizedown(const cv::cuda::GpuMat &umat, cv::cuda::GpuMat &lmat, const cv::Size &ofs);
void resizeup(const cv::cuda::GpuMat &lmat, cv::cuda::GpuMat &umat, const cv::Size &ofs = cv::Size(0, 0));
void shrink_opencv(struct_level* upper, struct_level* lower, const std::vector<cv::cuda::GpuMat> &umat, std::vector<cv::cuda::GpuMat> &lmat);
void hps_opencv(struct_level* upper, struct_level* lower, std::vector<cv::cuda::GpuMat> &umat, const std::vector<cv::cuda::GpuMat> &lmat);
void collapse_opencv(const cv::cuda::GpuMat &lower, cv::cuda::GpuMat &upper);
void dither_opencv(cv::cuda::GpuMat &top, cv::cuda::GpuMat &out);
#endif

void copy_channel(int i, int c);
void copy_channel_opencv(int i);
void mask_into_output(struct_level* input, float* mask, struct_level* output, bool first);
void mask_into_output_opencv(int i, int l, bool first);
void collapse(struct_level* lower, struct_level* upper);

void dither(struct_level* top, void* channel);
void blend();

//write
void jpeg_out();
void tiff_out();
void tiff_cvout();
void opencv_out();

//pseudowrap
void pseudowrap_split();
void pseudowrap_seam();
void pseudowrap_unsplit();

//go
void go();

//multiblend
void help();
void parse(std::vector<std::string> &output, const std::string &input);

#endif
