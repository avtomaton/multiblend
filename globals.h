#ifndef GLOBALS_H
#define GLOBALS_H

#include "structs.h"
#include <opencv2/cudaarithm.hpp>

typedef uint8_t mask_t;
extern int max_mask_value;

extern void* g_line0;
extern void* g_line1;
extern void* g_line2;
extern void* g_linet;
extern void* g_temp;
extern int* g_dither;
extern int g_numthreads;
extern int g_numimages;
extern int g_workwidth;
extern int g_workheight;
extern int g_workbpp;
extern int g_workbpp_cmd;
extern int g_min_top;
extern int g_min_left;
extern double g_xres;
extern double g_yres;
extern int g_levels;
extern int g_max_levels;
extern int g_sub_levels;
extern int g_verbosity;
extern bool g_wideblend;
extern bool g_seamwarning;
extern bool g_simpleseam;
extern bool g_reverse;
extern bool g_pseudowrap;
extern bool g_swap;
extern bool g_save_out_pyramids;
extern bool g_dewhorl;
extern char* g_output_filename;
extern char* g_seamload_filename;

extern png_color* g_palette;

extern char* g_seamsave_filename;
extern char* g_xor_filename;
extern int g_numchannels;
extern void** g_out_channels;
#if TIFF_LIBRARY
extern TIFF* g_tiff;
#endif
extern FILE* g_jpeg;
extern int g_compression;
extern int g_jpegquality;
extern uint32* g_seams;
extern bool g_timing;
extern bool g_savemasks;
extern bool g_nooutput;
extern bool g_caching;
//extern void* g_cache;
extern size_t g_cache_bytes;
extern struct_level* g_output_pyramid;
extern bool g_crop;
extern bool g_debug;
extern bool g_nomask;
extern bool g_bigtiff;
extern bool g_bgr;
extern uint32* g_edt;
extern struct_image* g_images;

#ifdef NO_CUDA
extern cv::Mat g_cvseams;
extern cv::Mat g_cvoutmask;
extern std::vector<std::vector<cv::Mat> > g_cvmaskpyramids;
extern std::vector<cv::Mat> g_cvmatpyramids;
extern std::vector<cv::Mat> g_cvmats;
extern std::vector<cv::Mat> g_cvmasks;
extern std::vector<cv::Mat> g_cvoutput_pyramid;
#else
extern cv::cuda::GpuMat g_cvseams;
extern cv::cuda::GpuMat g_cvoutmask;
extern std::vector<std::vector<cv::cuda::GpuMat> > g_cvmaskpyramids;
extern std::vector<std::vector<cv::cuda::GpuMat> > g_cvchannelpyramids;
extern std::vector<std::vector<cv::cuda::GpuMat> > g_cvchannels;
extern std::vector<cv::cuda::GpuMat> g_cvmasks;
extern std::vector<std::vector<cv::cuda::GpuMat> > g_cvoutput_channelpyramid;
extern std::vector<cv::Size> g_offsets;
extern std::vector<cv::Size> g_sizes;
#endif
extern cv::Mat g_cvout;

#endif
