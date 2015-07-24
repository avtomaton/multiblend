#include "structs.h"
#include "globals.h"
#include "functions.h"
#include "defines.h"

#include <algorithm>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>

void trim8(void* bitmap, uint32 w, uint32 h, int bpp, int* top, int* left, int* bottom, int* right) {
	size_t p;
	int x,y;
	uint32* b=(uint32*)bitmap;

// find first solid pixel
	x=0; y=0; p=0;
	while (p<(w*h)) {
		if (b[p++]>0x00ffffff) { // or maybe 0x00ffffff
			*top=y;
			*left=x;
			break;
		}
		x++; if (x==w) { x=0; y++; }
	}

// find last solid pixel
	x=w-1; y=h-1; p=w*h-1;
	while (p>=0) {
		if (b[p--]>0x00ffffff) {
			*bottom=y;
			*right=x;
			break;
		}
		x--; if (x<0) { x=w-1; y--; }
	}

	b+=*top*w;
	for (y=*top; y<=*bottom; y++) {
		for (x=0; x<*left; x++) {
			if (b[x]>0x00ffffff) {
				*left=x;
				break;
			}
		}
		for (x=w-1; x>*right; x--) {
			if (b[x]>0x00ffffff) {
				*right=x;
				break;
			}
		}
		b+=w;
	}
}

void trim16(void* bitmap, uint32 w, uint32 h, int bpp, int* top, int* left, int* bottom, int* right) {
	size_t p;
	int x,y;
	uint16* b=(uint16*)bitmap;

// find first solid pixel
	x=0; y=0; p=3;
	while (p<(w*h)) {
		if (b[p]!=0x0000) { // or maybe 0x00ffffff
			*top=y;
			*left=x;
			break;
		}
		p+=4;
		x++; if (x==w) { x=0; y++; }
	}

// find last solid pixel
	x=w-1; y=h-1; p=w*h*4-1;
	while (p>=0) {
		if (b[p]!=0x0000) {
			*bottom=y;
			*right=x;
			break;
		}
		p-=4;
		x--; if (x<0) { x=w-1; y--; }
	}

	b+=*top*w*4;
	for (y=*top; y<=*bottom; y++) {
		for (x=0; x<*left; x++) {
			if (b[x*4+3]!=0x0000) {
				*left=x;
				break;
			}
		}
		for (x=w-1; x>*right; x--) {
			if (b[x*4+3]!=0x0000) {
				*right=x;
				break;
			}
		}
		b+=w*4;
	}
}

void trim(void* bitmap, int w, int h, int bpp, int* top, int* left, int* bottom, int* right) {
	if (bpp==8) trim8(bitmap,w,h,bpp,top,left,bottom,right); else trim16(bitmap,w,h,bpp,top,left,bottom,right);
}

void extract8(struct_image* image, void* bitmap) {
	int x,y;
	size_t p,up;
	int mp=0;
	int masklast=-1,maskthis;
	int maskcount=0;
	size_t temp;
	uint32 pixel;

	image->binary_mask.rows=(uint32*)malloc((image->height+1)*sizeof(uint32));

	up=image->top*image->tiff_width+image->left;

	p=0;
	for (y=0; y<image->height; y++) {
		image->binary_mask.rows[y]=mp;

		pixel=((uint32*)bitmap)[up++];
		if (pixel>0xfeffffff) { // pixel is solid
			((uint8*)image->channels[0].data)[p]=pixel&0xff;
			((uint8*)image->channels[1].data)[p]=(pixel>>8)&0xff;
			((uint8*)image->channels[2].data)[p]=(pixel>>16)&0xff;
			masklast=1;
		} else {
			masklast=0;
		}
		maskcount=1;
		p++;

		for (x=1; x<image->width; x++) {
			pixel=((uint32*)bitmap)[up++];
			if (pixel>0xfeffffff) { // pixel is solid
				((uint8*)image->channels[0].data)[p]=pixel&0xff;
				((uint8*)image->channels[1].data)[p]=(pixel>>8)&0xff;
				((uint8*)image->channels[2].data)[p]=(pixel>>16)&0xff;
				maskthis=1;
			} else maskthis=0;
			if (maskthis!=masklast) {
				((uint32*)bitmap)[mp++]=masklast<<31|maskcount;
				masklast=maskthis;
				maskcount=1;
			} else maskcount++;
			p++;
		}
		((uint32*)bitmap)[mp++]=masklast<<31|maskcount;
		up+=image->tiff_width-image->width;
	}
	image->binary_mask.rows[y]=mp;

	image->binary_mask.data=(uint32*)malloc(mp * sizeof(uint32));
	temp=mp;

	memcpy(image->binary_mask.data,bitmap,mp * sizeof(uint32));
}

void extract_opencv(const cv::Mat &mask, const cv::Mat &channels, struct_image* image, void* bitmap)
{
	int x, y;
	size_t p;
	int mp = 0;
	int masklast = -1, maskthis;
	int maskcount = 0;

	image->binary_mask.rows = (uint32*)malloc((image->height + 1) * sizeof(uint32));

	p = 0;
	for (y = 0; y < image->height; ++y) {
		x = 0;
		image->binary_mask.rows[y] = mp;

		const uint8_t *pmask = mask.ptr<uint8_t>(y + image->ypos) + (x + image->xpos);
		const cv::Vec3b *pmat = channels.ptr<cv::Vec3b>(y + image->ypos) + (x + image->xpos);

		((uint8*)image->channels[0].data)[p] = (*pmat)[2];
		((uint8*)image->channels[1].data)[p] = (*pmat)[1];
		((uint8*)image->channels[2].data)[p] = (*pmat)[0];

		if (*pmask)  // pixel is solid
		{
			masklast = 1;
		}
		else
			masklast = 0;

		maskcount = 1;
		p++;
		++pmask;
		++pmat;

		for (x = 1; x < image->width; x++, ++pmask, ++pmat) {
			((uint8*)image->channels[0].data)[p] = (*pmat)[2];
			((uint8*)image->channels[1].data)[p] = (*pmat)[1];
			((uint8*)image->channels[2].data)[p] = (*pmat)[0];

			if (*pmask) // pixel is solid
			{
				maskthis = 1;
			}
			else
				maskthis = 0;
			if (maskthis != masklast)
			{
				((uint32*)bitmap)[mp++] = masklast<<31|maskcount;
				masklast = maskthis;
				maskcount = 1;
			}
			else
				maskcount++;
			p++;
		}
		((uint32*)bitmap)[mp++] = masklast<<31|maskcount;
	}
	image->binary_mask.rows[y] = mp;

	image->binary_mask.data = (uint32*)malloc(mp * sizeof(uint32));
	memcpy(image->binary_mask.data,bitmap,mp * sizeof(uint32));
}

void extract16(struct_image* image, void* bitmap) {
	int x,y;
	size_t p,up;
	int mp=0;
	int masklast=-1,maskthis;
	int maskcount=0;
	size_t temp;
	int mask;

	image->binary_mask.rows=(uint32*)malloc((image->height+1)*sizeof(uint32));

	up=(image->top*image->tiff_width+image->left)*4;

	p=0;
	for (y=0; y<image->height; y++) {
		image->binary_mask.rows[y]=mp;

		mask=((uint16*)bitmap)[up+3];
		if (mask==0xffff) { // pixel is 100% opaque
			((uint16*)image->channels[0].data)[p]=((uint16*)bitmap)[up+2];
			((uint16*)image->channels[1].data)[p]=((uint16*)bitmap)[up+1];
			((uint16*)image->channels[2].data)[p]=((uint16*)bitmap)[up];
			masklast=1;
		} else {
			masklast=0;
		}
		maskcount=1;

		up+=4;
		p++;

		for (x=1; x<image->width; x++) {
			mask=((uint16*)bitmap)[up+3];
			if (mask==0xffff) { // pixel is 100% opaque
				((uint16*)image->channels[0].data)[p]=((uint16*)bitmap)[up+2];
				((uint16*)image->channels[1].data)[p]=((uint16*)bitmap)[up+1];
				((uint16*)image->channels[2].data)[p]=((uint16*)bitmap)[up];
				maskthis=1;
			} else {
				maskthis=0;
			}
			if (maskthis!=masklast) {
				((uint32*)bitmap)[mp++]=masklast<<31|maskcount;
				masklast=maskthis;
				maskcount=1;
			} else maskcount++;

			up+=4;
			p++;
		}
		((uint32*)bitmap)[mp++]=masklast<<31|maskcount;
		up+=(image->tiff_width-image->width)*4;
	}
	image->binary_mask.rows[y]=mp;

	image->binary_mask.data=(uint32*)malloc(mp<<2);
	temp=mp;

	memcpy(image->binary_mask.data,bitmap,mp<<2);
}

void extract(struct_image* image, void* bitmap) {

	if (image->bpp==8) extract8(image, bitmap); else extract16(image, bitmap);
}

void to_cvmat(cv::Mat &mat, struct_image* image)
{
	int p = 0;
	for (int y = 0; y < image->height; ++y) {
		cv::Vec3b *pmat = mat.ptr<cv::Vec3b>(y + image->ypos) + (image->xpos);
		for (int x = 0; x < image->width; ++x, ++pmat, ++p)
		{
			(*pmat)[2] = ((uint8*)image->channels[0].data)[p];
			(*pmat)[1] = ((uint8*)image->channels[1].data)[p];
			(*pmat)[0] = ((uint8*)image->channels[2].data)[p];
		}
	}
}

#define NEXTMASK { mask=*mask_pointer++; maskcount=mask&0x7fffffff; mask=mask>>31; }
#define PREVMASK { mask=*--mask_pointer; maskcount=mask&0x7fffffff; mask=mask>>31; }

void inpaint8(struct_image* image, uint32* edt) {
	int x,y;
	int c;

	uint32* edt_p=edt;
	uint32* mask_pointer=image->binary_mask.data;
	int maskcount,mask;
	uint32 dist,temp_dist;
	int copy,temp_copy;
	uint8** chan_pointers=(uint8**)malloc(g_numchannels*sizeof(uint8*));
	int* p=(int*)malloc(g_numchannels*sizeof(int));
	bool lastpixel;

	for (c=0; c<g_numchannels; c++) chan_pointers[c]=(uint8*)image->channels[c].data;

// top-left to bottom-right
// first line, first block
	x=0;

	NEXTMASK;
	dist=(1-mask)<<31;
	for (; maskcount>0; maskcount--) edt_p[x++]=dist;

// first line, remaining blocks in first row
	while (x<image->width) {
		NEXTMASK;
		if (mask) {
			for (; maskcount>0; maskcount--) edt_p[x++]=0;
		} else { // mask if off, so previous mask must have been on
			dist=0;
			for (c=0; c<g_numchannels; c++) p[c]=chan_pointers[c][x-1];
			for (; maskcount>0; maskcount--) {
				dist+=2;
				for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=p[c];
				edt_p[x++]=dist;
			}
		}
	}

	for (y=image->height; y>1; y--) {
		lastpixel=false;
		edt_p+=image->width;
		for (c=0; c<g_numchannels; c++) chan_pointers[c]+=image->width;
		x=0;

		while (x<image->width) {
			NEXTMASK;
			if (mask) {
				for (; maskcount>0; maskcount--) edt_p[x++]=0;
			} else {
				if (x==0) {
					copy=x-image->width+1;
					dist=edt_p[copy]+3;

					temp_copy=x-image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
					edt_p[x++]=dist;
					maskcount--;
				}
				if (x+maskcount==image->width) {
					lastpixel=true;
					maskcount--;
				}

				for (; maskcount>0; maskcount--) {
					dist=edt_p[x-1]+2;
					copy=x-1;

					temp_copy=x-image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x-image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x-image->width+1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (dist<0x10000000) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
					}
					edt_p[x++]=dist; // dist
				}
				if (lastpixel) {
					dist=edt_p[x-1]+2;
					copy=x-1;

					temp_copy=x-image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x-image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (dist<0x10000000) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
					}
					edt_p[x++]=dist;
				}
			}
		}
	}

// bottom-right to top-left
	// last line

	while (x>0) {
		PREVMASK;
		if (mask) {
			x-=maskcount;
		} else {
			if (x==image->width) {
				x--;
				maskcount--;
			}
			for (c=0; c<g_numchannels; c++) p[c]=chan_pointers[c][x];
			for (; maskcount>0; maskcount--) {
				dist=edt_p[x]+2;
				x--;
				if (dist<edt_p[x]) {
					edt_p[x]=dist;
					for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=p[c];
				} else {
					for (c=0; c<g_numchannels; c++) p[c]=chan_pointers[c][x];
				}
			}
		}
	}

// remaining lines
	for (y=image->height; y>1; y--) {
		lastpixel=false;
		edt_p-=image->width;
		for (c=0; c<g_numchannels; c++) chan_pointers[c]-=image->width;
		x=image->width-1;

		while (x>=0) {
			PREVMASK;
			if (mask) {
				x-=maskcount;
			} else {
				if (x==image->width-1) {
					dist=edt_p[x];
					copy=0;

					temp_copy=x+image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (copy!=0) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
						edt_p[x--]=dist;
					} else x--;
					maskcount--;
				}
				if (x-maskcount==-1) {
					lastpixel=true;
					maskcount--;
				}
				for (; maskcount>0; maskcount--) {
					dist=edt_p[x];
					copy=0;

					temp_copy=x+1;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width+1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (copy!=0) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
						edt_p[x--]=dist;
					} else x--;
				}
				if (lastpixel) {
					dist=edt_p[x];
					copy=0;

					temp_copy=x+1;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width+1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (copy!=0) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
						edt_p[x--]=dist;
					} else x--;
				}
			}
		}
	}
	free(p);
	free(chan_pointers);
}

void inpaint16(struct_image* image, uint32* edt) {
	int x,y;
	int c;
	uint32* edt_p=edt;
	uint32* mask_pointer=image->binary_mask.data;
	int maskcount,mask;
	uint32 dist,temp_dist;
	int copy,temp_copy;
	uint16** chan_pointers=(uint16**)malloc(g_numchannels*sizeof(uint16*));
	int* p=(int*)malloc(g_numchannels*sizeof(int));
	bool lastpixel;

	for (c=0; c<g_numchannels; c++) chan_pointers[c]=(uint16*)image->channels[c].data;

// top-left to bottom-right
// first line, first block
	x=0;

	NEXTMASK;
	dist=(1-mask)<<31;
	for (; maskcount>0; maskcount--) edt_p[x++]=dist;

// first line, remaining blocks in first row
	while (x<image->width) {
		NEXTMASK;
		if (mask) {
			for (; maskcount>0; maskcount--) edt_p[x++]=0;
		} else { // mask if off, so previous mask must have been on
			dist=0;
			for (c=0; c<g_numchannels; c++) p[c]=chan_pointers[c][x-1];
			for (; maskcount>0; maskcount--) {
				dist+=2;
				for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=p[c];
				edt_p[x++]=dist;
			}
		}
	}

	for (y=image->height; y>1; y--) {
		lastpixel=false;
		edt_p+=image->width;
		for (c=0; c<g_numchannels; c++) chan_pointers[c]+=image->width; //p[c]+=image->width;
		x=0;

		while (x<image->width) {
			NEXTMASK;
			if (mask) {
				for (; maskcount>0; maskcount--) edt_p[x++]=0;
			} else {
				if (x==0) {
					copy=x-image->width+1;
					dist=edt_p[copy]+3;

					temp_copy=x-image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
					edt_p[x++]=dist;
					maskcount--;
				}
				if (x+maskcount==image->width) {
					lastpixel=true;
					maskcount--;
				}

				for (; maskcount>0; maskcount--) {
					dist=edt_p[x-1]+2;
					copy=x-1;

					temp_copy=x-image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x-image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x-image->width+1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (dist<0x10000000) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
					}
					edt_p[x++]=dist; // dist
				}
				if (lastpixel) {
					dist=edt_p[x-1]+2;
					copy=x-1;

					temp_copy=x-image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x-image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (dist<0x10000000) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
					}
					edt_p[x++]=dist;
				}
			}
		}
	}

// bottom-right to top-left
	// last line

	while (x>0) {
		PREVMASK;
		if (mask) {
			x-=maskcount;
		} else {
			if (x==image->width) {
				x--;
				maskcount--;
			}
			for (c=0; c<g_numchannels; c++) p[c]=chan_pointers[c][x];
			for (; maskcount>0; maskcount--) {
				dist=edt_p[x]+2;
				x--;
				if (dist<edt_p[x]) {
					edt_p[x]=dist;
					for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=p[c];
				} else {
					for (c=0; c<g_numchannels; c++) p[c]=chan_pointers[c][x];
				}
			}
		}
	}

// remaining lines
	for (y=image->height; y>1; y--) {
		lastpixel=false;
		edt_p-=image->width;
		for (c=0; c<g_numchannels; c++) chan_pointers[c]-=image->width;
		x=image->width-1;

		while (x>=0) {
			PREVMASK;
			if (mask) {
				x-=maskcount;
			} else {
				if (x==image->width-1) {
					dist=edt_p[x];
					copy=0;

					temp_copy=x+image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (copy!=0) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
						edt_p[x--]=dist;
					} else x--;
					maskcount--;
				}
				if (x-maskcount==-1) {
					lastpixel=true;
					maskcount--;
				}
				for (; maskcount>0; maskcount--) {
					dist=edt_p[x];
					copy=0;

					temp_copy=x+1;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width+1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width-1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (copy!=0) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
						edt_p[x--]=dist;
					} else x--;
				}
				if (lastpixel) {
					dist=edt_p[x];
					copy=0;

					temp_copy=x+1;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width+1;
					temp_dist=edt_p[temp_copy]+3;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					temp_copy=x+image->width;
					temp_dist=edt_p[temp_copy]+2;
					if (temp_dist<dist) {
						dist=temp_dist;
						copy=temp_copy;
					}

					if (copy!=0) {
						for (c=0; c<g_numchannels; c++) chan_pointers[c][x]=chan_pointers[c][copy];
						edt_p[x--]=dist;
					} else x--;
				}
			}
		}
	}
	free(p);
	free(chan_pointers);
}

void inpaint(struct_image* image, uint32* edt) {
	if (image->bpp==8) inpaint8(image,edt); else inpaint16(image,edt);
}

#ifdef NO_CUDA
void init_dist(const cv::Mat &mask, cv::Mat &dist, struct_image* image)
{
	for (int y = image->ypos; y < (image->ypos + image->height); ++y)
	{
		const uint8_t *pmask = mask.ptr<uint8_t>(y);
		int *pdist = dist.ptr<int>(y);
		for (int x = image->xpos; x < (image->xpos + image->width); ++x)
		{
			if (pmask[x])
				pdist[x] = 0;
			else
				pdist[x] = image->height + image->width;
		}
	}
}
#else
void init_dist(const cv::cuda::GpuMat &mask, cv::cuda::GpuMat &dist, struct_image* image)
{
	printf("init_dist(cuda)\n");
	exit(1);
}
#endif

#ifdef NO_CUDA
void find_distances_cycle_y_horiz(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight)
{
	const uint8_t* pmask = NULL;
	int* pdist = NULL;
	cv::Vec3b* pmat = NULL;

	for (int y = ybeg; y < yend; ++y)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<cv::Vec3b>(y);
		pmask = mask.ptr<uint8_t>(y);
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
}
#else
void find_distances_cycle_y_horiz(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight)
{
	printf("find_distances_cycle_y_horiz(cuda)\n");
	exit(1);
}
#endif

#ifdef NO_CUDA
inline void find_distances_cycle_x(
	const uint8_t *pmask, int *pdist, int *pdist_prev, cv::Vec3b *pnums, cv::Vec3b *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag)
{
	for (int x = tmp_xbeg; x < tmp_xend; ++x)
	{
		if (pmask[x] || pdist[x] == 0)
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
#else
inline void find_distances_cycle_x(
	const uint8_t *pmask, int *pdist, int *pdist_prev, cv::Vec3b *pnums, cv::Vec3b *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag)
{
	printf("find_distances_cycle_x(cuda)\n");
	exit(1);
}
#endif

#ifdef NO_CUDA
void find_distances_cycle_y_vert(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend, int xl, int xr,
	bool two_areas,
	int l_straight, int l_diag)
{
	const uint8_t *pmask = NULL;
	int *pdist = NULL;
	cv::Vec3b *pmat = NULL;
	auto pdist_prev = dist.ptr<int>(ybeg - shift);
	auto pmat_prev = mat.ptr<cv::Vec3b>(ybeg - shift);

	int y = ybeg;
	while (y != yend)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<cv::Vec3b>(y);
		pmask = mask.ptr<uint8_t>(y);

		if (two_areas)
		{
			find_distances_cycle_x(pmask, pdist, pdist_prev, pmat, pmat_prev, xbeg, xr, l_straight, l_diag);
			find_distances_cycle_x(pmask, pdist, pdist_prev, pmat, pmat_prev, xl, xend, l_straight, l_diag);
		}
		else
		{
			find_distances_cycle_x(pmask, pdist, pdist_prev, pmat, pmat_prev, xbeg, xend, l_straight, l_diag);
		}

		pdist_prev = pdist;
		pmat_prev = pmat;

		y += shift;
	}
}
#else
void find_distances_cycle_y_vert(
	cv::cuda::GpuMat &dist, cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask,
	int shift, int ybeg, int yend, int xbeg, int xend, int xl, int xr,
	bool two_areas,
	int l_straight, int l_diag)
{
	printf("find_distances_cycle_y_vert\n");
	exit(1);
}
#endif

#ifdef NO_CUDA
void inpaint_opencv(cv::Mat &mat, const cv::Mat &mask, struct_image* image, cv::Mat &dist)
#else
void inpaint_opencv(cv::cuda::GpuMat &mat, const cv::cuda::GpuMat &mask, struct_image* image, cv::cuda::GpuMat &dist)
#endif
{
	init_dist(mask, dist, image);

	int xl = image->xpos, xr = (image->xpos + image->width);
	bool two_areas = is_two_areas(mask, image);
	if (two_areas)
	{
		xl = search_l(mask, image->xpos + image->width / 2, image->xpos + image->width, false);
		xr = search_r(mask, image->xpos, image->xpos + image->width / 2, false) + 1;
		//printf("xl = %d, xr = %d\n", xl, xr);
	}

	int ybeg, yend;
	int xbeg, xend;

//vertical
	xbeg = image->xpos;
	xend = image->xpos + image->width;

	// top to bottom
	ybeg = image->ypos + 1;
	yend = image->ypos + image->height;

	find_distances_cycle_y_vert(dist, mat, mask, 1, ybeg, yend, xbeg, xend, xl, xr, two_areas, L_STRAIGHT, L_DIAG);

	// bottom to top
	ybeg = image->ypos + image->height - 1 - 1;
	yend = image->ypos - 1;

	find_distances_cycle_y_vert(dist, mat, mask, -1, ybeg, yend, xbeg, xend, xl, xr, two_areas, L_STRAIGHT, L_DIAG);

//horizontal
	ybeg = image->ypos;
	yend = image->ypos + image->height;

	//left to right
	xbeg = image->xpos + 1;
	xend = xr;
	find_distances_cycle_y_horiz(dist, mat, mask, 1, ybeg, yend, xbeg, xend, L_STRAIGHT);

	//right to left
	xbeg = (image->xpos + image->width - 1) - 1;
	xend = xl - 1;

	find_distances_cycle_y_horiz(dist, mat, mask, -1, ybeg, yend, xbeg, xend, L_STRAIGHT);
}

void tighten() {

	int i;
	int max_right=0,max_bottom=0;

	g_min_left=0x7fffffff;
	g_min_top=0x7fffffff;

	for (i=0; i<g_numimages; i++) {
		g_min_left=std::min(g_min_left,g_images[i].xpos);
		g_min_top=std::min(g_min_top,g_images[i].ypos);
	}

	for (i=0; i<g_numimages; i++) {
		g_images[i].xpos-=g_min_left;
		g_images[i].ypos-=g_min_top;
	}

	for (i=0; i<g_numimages; i++) {
		max_right=std::max(max_right,g_images[i].xpos+g_images[i].width);
		max_bottom=std::max(max_bottom,g_images[i].ypos+g_images[i].height);
	}

	g_workwidth=max_right;
	g_workheight=max_bottom;
}

int localize_xl(const cv::Mat &mask, float j0, float jstep, float left, float right)
{
	for (float j = left + j0; j < right; j += jstep)
	{
		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uint8_t>(i, (int)j))
				return (int)j;
		}
	}
	return (int)right;
}

int localize_xr(const cv::Mat &mask, float j0, float jstep, float left, float right)
{
	for (float j = right - j0; j > left; j -= jstep)
	{
		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uint8_t>(i, (int)j))
				return (int)j;
		}
	}
	return (int)left;
}

int localize_yl(const cv::Mat &mask, float i0, float istep, float left, float right)
{
	for (float i = left + i0; i < right; i += istep)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uint8_t>((int)i, j))
				return (int)i;
		}
	}
	return (int)right;
}

int localize_yr(const cv::Mat &mask, float i0, float istep, float left, float right)
{
	for (float i = right - i0; i > left; i -= istep)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uint8_t>((int)i, j))
				return (int)i;
		}
	}
	return (int)left;
}

#ifdef NO_CUDA
int search_l(const cv::Mat &mask, float left, float right, bool isy)
{
	int l;
	float i0;
	float factor0 = 4.0f;
	float factor1 = 4.0f;

	float istep = (right - left) / factor0;

	while (abs((int)right - (int)left) > 1)
	{
		float istep0 = istep;
		i0 = istep;
		if (isy)
			l = localize_yl(mask, i0, istep, left, right);
		else
			l = localize_xl(mask, i0, istep, left, right);

		while ((l == right) && istep >= 0.5)
		{
			i0 = istep / 2;
			if (isy)
				l = localize_yl(mask, i0, istep, left, right);
			else
				l = localize_xl(mask, i0, istep, left, right);

			istep /= 2;
		}

		right = (float)l;
		left = right - istep;
		istep = (right - left) / factor1;
	}

	return l;
}
#else
int search_l(const cv::cuda::GpuMat &mask, float left, float right, bool isy)
{
	printf("search_l\n");
	exit(1);
	return 0;
}
#endif

#ifdef NO_CUDA
int search_r(const cv::Mat &mask, float left, float right, bool isy)
{
	int r;
	float i0;
	float factor0 = 4.0f;
	float factor1 = 4.0f;
	float istep = (right - left) / factor0;

	while (abs((int)right - (int)left) > 1)
	{
		i0 = istep;
		if (isy)
			r = localize_yr(mask, i0, istep, left, right);
		else
			r = localize_xr(mask, i0, istep, left, right);

		while (r == left && istep >= 0.5)
		{
			i0 = istep / 2;
			if (isy)
				r = localize_yr(mask, i0, istep, left, right);
			else 
				r = localize_xr(mask, i0, istep, left, right);

			istep /= 2;
		}

		left = (float)r;
		right = left + istep;

		istep = (right - left) / factor1;
	}
	return r;
}
#else
int search_r(const cv::cuda::GpuMat &mask, float left, float right, bool isy)
{
	printf("search_r\n");
	exit(1);
	return 0;
}
#endif

#ifdef NO_CUDA
cv::Rect get_visible_rect(const cv::Mat &mask)
{
	int xl = mask.cols, yl = mask.rows, xr = -1, yr = -1;
	
	int boundary_strip = 2;
	float left, right;
	//try top boundary
	for (int i = 0; i < boundary_strip && i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uint8_t>(i, j))
			{
				yl = i;
				i = mask.rows;
				break;
			}
		}
	}

	//try bottom boundary
	for (int i = mask.rows - 1; (i >= mask.rows - boundary_strip) && (i >= 0); --i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uint8_t>(i, j))
			{
				yr = i;
				i = - 1;
				break;
			}
		}
	}

	if (yl == mask.rows)
	{
		//top
		left = 0;
		right = (float)mask.rows;
		yl = search_l(mask, left, right, true);
		if (yl == mask.rows)
			die("yl == mask.rows: no visible pixels");
	}

	if (yr == -1)
	{
		//bottom
		left = (float)yl;
		right = (float)mask.rows;
		yr = search_r(mask, left, right, true);
		if (yr == -1)
			die("yr == -1: no visible pixels");
	}

	//try left boundary
	for (int j = 0; j < boundary_strip && j < mask.cols; ++j)
	{
		for (int i = yl; i <= yr; ++i)
		{
			if (mask.at<uint8_t>(i, j))
			{
				xl = j;
				j = mask.cols;
				break;
			}
		}
	}

	//try right boundary
	for (int j = mask.cols - 1; j >= mask.cols - boundary_strip && j >= 0; --j)
	{
		for (int i = yl; i <= yr; ++i)
		{
			if (mask.at<uint8_t>(i, j))
			{
				xr = j;
				j = - 1;
				break;
			}
		}
	}

	if (xl == mask.cols)
	{
		//left
		left = 0;
		right = (float)mask.cols;
		xl = search_l(mask, left, right, false);
		if (xl == mask.cols)
			die("xl == mask.cols: no visible pixels");
	}
	if (xr == -1)
	{
		//right
		left = (float)xl;
		right = (float)mask.cols;
		xr = search_r(mask, left, right, false);
		if (xr == -1)
			die("xr == -1: no visible pixels");
	}
	/*
	int xl2 = mask.cols, yl2 = mask.rows, xr2 = -1, yr2 = -1;
	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uint8_t>(i, j))
			{
				if (xl2 > j)
					xl2 = j;
				if (yl2 > i)
					yl2 = i;
				if (xr2 < j)
					xr2 = j;
				if (yr2 < i)
					yr2 = i;
			}
		}
	}

	if (xl != xl2)
		die("xl = %d, xl2 = %d\n", xl, xl2);
	if (xr != xr2)
		die("xr = %d, xr2 = %d\n", xr, xr2);
	if (yl != yl2)
		die("yl = %d, yl2 = %d\n", yl, yl2);
	if (yr != yr2)
		die("yr = %d, yr2 = %d\n", yr, yr2);
	*/
	cv::Rect res;
	res.x = xl;
	res.y = yl;
	res.width = xr - xl + 1;
	res.height = yr - yl + 1;
	return res;
}
#else
cv::Rect get_visible_rect(const cv::cuda::GpuMat &mask)
{
	cv::Rect res;
	printf("get_visible_rect(cuda)\n");
	exit(1);
	return res;
}
#endif

#ifdef NO_CUDA
void mat2struct(int i, const std::string &filename, cv::Mat &matimage, const cv::Mat &mask, cv::Mat &dist)
#else
void mat2struct(int i, const std::string &filename, cv::cuda::GpuMat &matimage, const cv::cuda::GpuMat &mask, cv::cuda::GpuMat &dist)
#endif
{
	#ifdef WIN32
		strcpy_s(g_images[i].filename, 256, filename.c_str());
	#else
		strncpy(g_images[i].filename, filename.c_str(), 256);
	#endif

	cv::Rect vis_rect = get_visible_rect(mask);
	I.xpos = vis_rect.x;
	I.ypos = vis_rect.y;
	I.width = vis_rect.width;
	I.height = vis_rect.height;

	#ifdef NO_OPENCV
		for (int c = 0; c < g_numchannels; ++c)
		{
			if (!(g_images[i].channels[c].data = (void*)malloc((g_images[i].width * g_images[i].height) << (g_images[i].bpp >> 4))))
				die("not enough memory for image channel");
			//free(g_images[i].channels[c].data) <=> free(pixels) in void copy_channel(int i, int c), blending.cpp
		}
	#endif

	printf("vis_rect: %d, %d, %d, %d\n", vis_rect.x,vis_rect.y,vis_rect.width,vis_rect.height);

	g_workwidth = std::max(g_workwidth, (int)(I.xpos + I.width));
	g_workheight = std::max(g_workheight, (int)(I.ypos + I.height));
	
	inpaint_opencv(matimage, mask, &I, dist);

	#ifdef NO_OPENCV
		void* untrimmed = (void*)malloc(I.width * I.height * sizeof(uint32));
		if (!untrimmed) die("not enough memory to process images");
		extract_opencv(mask, matimage, &I, untrimmed);
		//cv::Mat inp_mat = matimage.clone();
		//inpaint(&I, (uint32*)untrimmed);
		//to_cvmat(inp_mat, &I);
		//std::string out_inpaint = std::string("J:\\git\\multiblend\\multiblend\\x64\\ReleaseApp\\") + std::to_string(i) + std::string("_after_inpaint.png");
		//cv::imwrite(out_inpaint, matimage);
		free(untrimmed);
	#endif
}

void load_images() {
	char buf[256];
	#ifdef NO_CUDA
	cv::Mat dist(g_cvmats[0].size(), CV_32S);
	#else
	cv::cuda::GpuMat dist(g_cvmats[0].size(), CV_32S);
	#endif

	for (int i = 0; i < g_numimages; ++i) {
		//cv::Mat matimage = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
		//cv::Mat mask = cv::imread(std::string("mask_") + std::string(argv[i]), CV_LOAD_IMAGE_GRAYSCALE);
		#if _WIN32
		sprintf_s(buf, "%d/", i);
		#else
		sprintf(buf, "%d/", i);
		#endif
		mat2struct(i, buf, g_cvmats[i], g_cvmasks[i], dist);
	}

	if (g_crop) tighten();

	if (g_workbpp_cmd!=0) {
		if (g_workbpp_cmd==16 && g_jpegquality!=-1) {
			output(0,"warning: JPEG output; overriding to 8bpp\n");
			g_workbpp_cmd=8;
		}
		g_workbpp=g_workbpp_cmd;
	} else {
		g_workbpp=g_images[0].bpp;
	}
}
