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

void clear_temp();
void die(const char* error, ...);
bool is_two_areas(const cv::Mat &mask, struct_image* image);

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

template<typename T>
void find_distances_cycle_x(
	const uint8_t *pmask, 
	int *pdist, 
	int *pdist_prev, 
	T *pnums, 
	T *pnums_prev, 
	int tmp_xbeg, 
	int tmp_xend, 
	bool invert_mask, 
	int l_straight, 
	int l_diag)
{
	for (int x = tmp_xbeg; x < tmp_xend; ++x)
	{
		if (invert_mask ? !pmask[x] : pmask[x])
			continue;
		if (pdist[x] == 0)
			continue;

		if (pdist_prev[x] + l_straight < pdist[x])
		{
			pdist[x] = pdist_prev[x] + l_straight;
			pnums[x] = pnums_prev[x];
		}

		if (x != tmp_xbeg)
		{
			if (pdist_prev[x - 1] + l_diag < pdist[x])
			{
				pdist[x] = pdist_prev[x - 1] + l_diag;
				pnums[x] = pnums_prev[x - 1];
			}
		}

		if (x != (tmp_xend - 1))
		{
			if (pdist_prev[x + 1] + l_diag < pdist[x])
			{
				pdist[x] = pdist_prev[x + 1] + l_diag;
				pnums[x] = pnums_prev[x + 1];
			}
		}


	}
}


template<typename T>
void find_seamdistances_cycle_x(
	const std::vector<const uint8_t*> &pmasks,
	int *pdist,
	int *pdist_prev,
	T *pnums,
	T *pnums_prev,
	int tmp_xbeg,
	int tmp_xend,
	int l_straight,
	int l_diag)
{
	for (int x = tmp_xbeg; x < tmp_xend; ++x)
	{
		if (pdist[x] == 0)
			continue;

		if (pdist_prev[x] + l_straight < pdist[x] && pmasks[pnums_prev[x]][x])
		{
			pdist[x] = pdist_prev[x] + l_straight;
			pnums[x] = pnums_prev[x];
		}

		if (x != tmp_xbeg)
		{
			if (pdist_prev[x - 1] + l_diag < pdist[x] && pmasks[pnums_prev[x-1]][x])
			{
				pdist[x] = pdist_prev[x - 1] + l_diag;
				pnums[x] = pnums_prev[x - 1];
			}
		}

		if (x != (tmp_xend - 1))
		{
			if (pdist_prev[x + 1] + l_diag < pdist[x] && pmasks[pnums_prev[x + 1]][x])
			{
				pdist[x] = pdist_prev[x + 1] + l_diag;
				pnums[x] = pnums_prev[x + 1];
			}
		}
	}
}

template<typename T>
void find_distances_cycle_y_horiz(
	cv::Mat &dist, 
	cv::Mat &mat, 
	const cv::Mat &mask, 
	int shift, 
	int ybeg, 
	int yend, 
	int xbeg, 
	int xend, 
	bool invert_mask, 
	int l_straight)
{
	const uint8_t* pmask = NULL;
	int* pdist = NULL;
	T* pmat = NULL;

	for (int y = ybeg; y < yend; ++y)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<T>(y);
		pmask = mask.ptr<uint8_t>(y);
		int x = xbeg;
		while (x != xend)
		{
			if (invert_mask ? !pmask[x] : pmask[x])
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
}

template<typename T>
void find_seamdistances_cycle_y_horiz(
	cv::Mat &dist,
	cv::Mat &mat,
	const std::vector<cv::Mat> &masks,
	int shift,
	int ybeg,
	int yend,
	int xbeg,
	int xend,
	int l_straight)
{
	std::vector<const uint8_t*> pmasks(masks.size(), NULL);
	int* pdist = NULL;
	T* pmat = NULL;

	for (int y = ybeg; y < yend; ++y)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<T>(y);
		for (int i = 0; i < g_numimages; ++i)
			pmasks[i] = masks[i].ptr<uint8_t>(y);
		int x = xbeg;
		while (x != xend)
		{
			if (pdist[x] == 0)
			{
				x += shift;
				continue;
			}
			if (pdist[x - shift] + l_straight < pdist[x] && pmasks[pmat[x - shift]][x])
			{
				pdist[x] = pdist[x - shift] + l_straight;
				pmat[x] = pmat[x - shift];
			}
			x += shift;
		}
	}
}

template<typename T>
void find_distances_cycle_y_vert(
	cv::Mat &dist, 
	cv::Mat &mat, 
	const cv::Mat &mask, 
	int shift, 
	int ybeg, 
	int yend, 
	int xbeg, 
	int xend, 
	int xl,
	int xr,
	bool two_areas, 
	bool invert_mask, 
	int l_straight,
	int l_diag)
{
	const uint8_t *pmask = NULL;
	int *pdist = NULL;
	T *pmat = NULL;
	int* pdist_prev = dist.ptr<int>(ybeg - shift);
	T* pmat_prev = mat.ptr<T>(ybeg - shift);

	int y = ybeg;
	while (y != yend)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<T>(y);
		pmask = mask.ptr<uint8_t>(y);
				
		if (two_areas)
		{
			find_distances_cycle_x<T>(pmask, pdist, pdist_prev, pmat, pmat_prev, xbeg, xr, invert_mask, l_straight, l_diag);
			find_distances_cycle_x<T>(pmask, pdist, pdist_prev, pmat, pmat_prev, xl, xend, invert_mask, l_straight, l_diag);
		}
		else
		{
			find_distances_cycle_x<T>(pmask, pdist, pdist_prev, pmat, pmat_prev, xbeg, xend, invert_mask, l_straight, l_diag);
		}

		pdist_prev = pdist;
		pmat_prev = pmat;

		y += shift;
	}
}


template<typename T>
void find_seamdistances_cycle_y_vert(
	cv::Mat &dist,
	cv::Mat &mat,
	const std::vector<cv::Mat> &masks,
	int shift,
	int ybeg,
	int yend,
	int xbeg,
	int xend,
	int l_straight,
	int l_diag)
{
	std::vector<const uint8_t*> pmasks(masks.size(), NULL);
	int *pdist = NULL;
	T *pmat = NULL;
	int* pdist_prev = dist.ptr<int>(ybeg - shift);
	T* pmat_prev = mat.ptr<T>(ybeg - shift);

	int y = ybeg;
	while (y != yend)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<T>(y);
		for (int i = 0; i < g_numimages; ++i)
			pmasks[i] = masks[i].ptr<uint8_t>(y);

		find_seamdistances_cycle_x<T>(pmasks, pdist, pdist_prev, pmat, pmat_prev, xbeg, xend, l_straight, l_diag);

		pdist_prev = pdist;
		pmat_prev = pmat;

		y += shift;
	}
}




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
void init_dist(const cv::Mat &mask, cv::Mat &dist, struct_image* image);

void inpaint_opencv(cv::Mat &mat, const cv::Mat &mask, struct_image* image, cv::Mat &dist);
void tighten();
int localize_xl(const cv::Mat &mask, float j0, float jstep, float left, float right);
int localize_xr(const cv::Mat &mask, float j0, float jstep, float left, float right);
int localize_yl(const cv::Mat &mask, float i0, float istep, float left, float right);
int localize_yr(const cv::Mat &mask, float i0, float istep, float left, float right);
int search_l(const cv::Mat &mask, float left, float right, bool isy);
int search_r(const cv::Mat &mask, float left, float right, bool isy);
cv::Rect get_visible_rect(const cv::Mat &mask);
void mat2struct(int i, const std::string &filename, cv::Mat &matimage, const cv::Mat &mask, cv::Mat &dist);
void load_images();

//seaming
void seam_png(int mode, const char* filename);
void load_seams();
void rightdownxy();
void leftupxy();
void simple_seam();
void make_seams();
void seam(cv::Mat &nums);

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
void copy_channel(int i, int c);
void mask_into_output(struct_level* input, float* mask, struct_level* output, bool first);
void collapse(struct_level* lower, struct_level* upper);
void dither(struct_level* top, void* channel);
void blend();

//write
void jpeg_out();
void tiff_out();
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
