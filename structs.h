#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <png.h>

#if JPEG_LIBRARY
#include <jpeglib.h>
#endif
#if TIFF_LIBRARY
#include <tiffio.h>
#endif

#ifdef WIN32
#define NOMINMAX
#include <Windows.h>
#undef min
#undef max
#endif

#ifdef __APPLE__
	#define memalign(a,b) malloc((b))
#else
	#include <malloc.h>
#endif

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef int16_t int16;
typedef int32_t int32;

union intfloat {
	float f;
	int i;
};

struct struct_indexed {
//  int size;
//  uint32 p;
	uint32* pointer;
	uint32* data;
	uint32* rows;
//  uint32* w;
//  uint32* h;
};

struct struct_level {
	void* data;
	size_t offset;
	int x0,y0; // inclusive minimum coordinates within this extended level
	int x1,y1; // inclusive maximum coordinates within this extended level
	int w,h;
	int pitch;
};

struct GeoTIFFInfo {
	double XGeoRef, YGeoRef;
	double XCellRes, YCellRes;
	double projection[16];
	int    nodata;
	bool   set;
};

struct struct_channel {
	void* data;
	char* filename;
	FILE* f;
};

struct struct_image {
	char filename[256];
	uint16 bpp;
	int width,height;
	int xpos,ypos;
	int top,left;
	struct_channel* channels;
	struct_indexed binary_mask;
	struct_level* pyramid;
	float** masks;
	bool seampresent;
	#if TIFF_LIBRARY
	GeoTIFFInfo geotiff;
	TIFF* tiff;
	#endif
	int tiff_width;
	int tiff_height;
	int tiff_u_height;
	int first_strip;
	int last_strip;

	int cx;
	int cy;
	int d,dx;
	void reset()
	{
		width = height = xpos = ypos = 0;
	}
};







#ifdef WIN32
class my_timer {
public:
	void set();
	double read();
	my_timer();
	void report(const char* name);
private:
	LARGE_INTEGER t1;
	LARGE_INTEGER t2;
	LARGE_INTEGER frequency;
};
#else
class my_timer {
public:
	void set();
	double read();
	void report(const char* name);
};
#endif

#endif
