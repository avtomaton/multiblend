#include "functions.h"
#include "globals.h"
#include "cuda-functions.h"
#include  <cstdarg>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/cudaarithm.hpp>

void print_gpu_memory()
{
	static int oldused = 0;
	size_t free, total;
	cuda_get_memory(&free, &total);
	printf("cuda memory: %f MB, %f MB\n", (total - free) / (1024.0*1024.0), free / (1024.0*1024.0));
	printf("diff = %f MB\n\n", ((int)(total - free) - oldused) / (1024.0*1024.0));
	oldused = total - free;
}


void clear_temp() {
	int i, c;

	for (i = 0; i<g_numimages; i++) {
		for (c = 0; c<g_numchannels; c++) {
			if (g_images[i].channels[c].f) {
				fclose(g_images[i].channels[c].f);
#ifdef WIN32
				DeleteFile(g_images[i].channels[c].filename);
#endif
			}
		}
	}
}

void clean_globals()
{
	#ifdef NO_OPENCV
	for (int c = 0; c < g_numchannels; ++c)
		_aligned_free(g_out_channels[c]);

	free(g_out_channels);

	for (int i = 0; i < g_numimages; ++i)
		for (int l = 0; l < g_levels; ++l)
			free(g_images[i].masks[l]);
	free(g_seams);

	for (int i = 0; i < g_numimages; ++i)
		free(g_images[i].masks);

	_aligned_free(g_line2);
	_aligned_free(g_line1);
	_aligned_free(g_line0);

	free(g_palette);

	for (int i = 0; i < g_numimages; ++i)
	{
		free(g_images[i].binary_mask.data);
		free(g_images[i].binary_mask.rows);
		free(g_images[i].channels);
	}

	#endif

	free(g_images);
}

void output(int level, const char* fmt, ...) {
	va_list args;

	if (level<=g_verbosity) {
		va_start(args,fmt);
		vprintf(fmt,args);
		va_end(args);
	}
}

void report_time(const char* name, double time) {
	if (g_timing) output(0,"%s: %.3fs\n",name,time);
}

#ifndef WIN32
#define SNPRINTF snprintf
int _stricmp(const char* a, const char* b) { return strcasecmp(a,b); }
void* _aligned_malloc(size_t size, int boundary) { return memalign(boundary,size); }
void _aligned_free(void* a) { free(a); }
void fopen_s(FILE** f,const char* filename, const char* mode) { *f=fopen(filename,mode); }
#else
#define SNPRINTF _snprintf_s
#endif

void die(const char* error, ...) {
	va_list args;

	va_start(args,error);
	vprintf(error,args);
	va_end(args);
	printf("\n");

	//clear_temp();

	if (g_debug) {
		printf("\nPress Enter to end\n");
		getchar();
	}

	exit(1);
}
