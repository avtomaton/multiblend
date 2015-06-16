#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "stdlib.h"
#include "stdio.h"

#include "structs.h"

#include <opencv2/core/core.hpp>

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

//geotiff
void geotiff_register(TIFF* tif);
int geotiff_read(TIFF* tiff, GeoTIFFInfo* info);
int geotiff_write(TIFF * tiff, GeoTIFFInfo * info);

//load images
void trim8(void* bitmap, uint32 w, uint32 h, int bpp, int* top, int* left, int* bottom, int* right);
void trim16(void* bitmap, uint32 w, uint32 h, int bpp, int* top, int* left, int* bottom, int* right);
void trim(void* bitmap, int w, int h, int bpp, int* top, int* left, int* bottom, int* right);
void extract8(struct_image* image, void* bitmap);
void extract16(struct_image* image, void* bitmap);
void extract(struct_image* image, void* bitmap);
void inpaint8(struct_image* image, uint32* edt);
void inpaint16(struct_image* image, uint32* edt);
void inpaint(struct_image* image, uint32* edt);
void tighten();
void load_images(const std::vector<cv::Mat> &mats, const std::vector<cv::Mat> &masks);

//seaming
void seam_png(int mode, const char* filename);
void load_seams();
void rightdownxy();
void leftupxy();
void simple_seam();
void make_seams();
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
void copy_channel(int i, int c);
void mask_into_output(struct_level* input, float* mask, struct_level* output, bool first);
void collapse(struct_level* lower, struct_level* upper);
void dither(struct_level* top, void* channel);
void blend();

//write
void jpeg_out();
void tiff_out();

//pseudowrap
void pseudowrap_split();
void pseudowrap_seam();
void pseudowrap_unsplit();

//go
void go(const std::vector<cv::Mat> &mats, const std::vector<cv::Mat> &masks);

//multiblend
void help();
void parse(std::vector<std::string> &output, const std::string &input);

#endif
