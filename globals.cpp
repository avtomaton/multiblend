#include "globals.h"


Profiler mprofiler;

void* g_line0 = NULL;
void* g_line1 = NULL;
void* g_line2 = NULL;
void* g_linet = NULL;

void* g_temp = NULL;

int* g_dither = NULL;

int g_numthreads;

int g_numimages=0;
int g_workwidth=0;
int g_workheight=0;
int g_workbpp=0;
int g_workbpp_cmd=0;
int g_min_top=0;
int g_min_left=0;
double g_xres=-1;
double g_yres=-1;
int g_levels;
int g_max_levels=1000000;
int g_sub_levels=0;
int g_verbosity=1;
bool g_wideblend=false;
bool g_seamwarning=false;
bool g_simpleseam=false;
bool g_reverse=false;
bool g_pseudowrap=false;
bool g_swap=false;
bool g_save_out_pyramids=false;
bool g_dewhorl=false;
char* g_output_filename = NULL;
char* g_seamload_filename = NULL;

png_color* g_palette = NULL;

char* g_seamsave_filename = NULL;
char* g_xor_filename = NULL;
int g_numchannels=3;
void** g_out_channels = NULL;
#if TIFF_LIBRARY
TIFF* g_tiff = NULL;
#endif
FILE* g_jpeg = NULL;
int g_compression=-1;
int g_jpegquality=-1;
uint32* g_seams = NULL;
bool g_timing=false;
bool g_savemasks=false;
bool g_nooutput=false;
bool g_caching=false;
//void* g_cache;
size_t g_cache_bytes=0;

struct_level* g_output_pyramid = NULL;

bool g_crop=true;
bool g_debug=false;
bool g_nomask=false;
bool g_bigtiff=false;
bool g_bgr=false;

uint32* g_edt = NULL;

struct_image* g_images = NULL;

#ifdef NO_CUDA
cv::Mat g_cvseams;
cv::Mat g_cvoutmask;
std::vector<std::vector<cv::Mat> > g_cvmaskpyramids;
std::vector<cv::Mat> g_cvmatpyramids;
std::vector<cv::Mat> g_cvmats;
std::vector<cv::Mat> g_cvmasks;
std::vector<cv::Mat> g_cvoutput_pyramid;
#else
cv::cuda::GpuMat g_cvseams;
cv::cuda::GpuMat g_cvoutmask;
std::vector<std::vector<cv::cuda::GpuMat> > g_cvmaskpyramids;
std::vector<cv::cuda::GpuMat> g_cvmatpyramids;
std::vector<cv::cuda::GpuMat> g_cvmats;
std::vector<cv::cuda::GpuMat> g_cvmasks;
std::vector<cv::cuda::GpuMat> g_cvoutput_pyramid;
#endif
cv::Mat g_cvout;
