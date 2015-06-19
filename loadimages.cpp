#include "structs.h"
#include "globals.h"
#include "functions.h"
#include "defines.h"

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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
	Proftimer proftimer(&mprofiler, "trim");
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

	Proftimer proftimer_extract_memcpy(&mprofiler, "extract_memcpy");
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

		if (mask.at<uint8_t>(y + image->ypos, x + image->xpos))  // pixel is solid
		{
			cv::Vec3b pix = channels.at<cv::Vec3b>(y + image->ypos, x + image->xpos);
			((uint8*)image->channels[0].data)[p] = pix[2];
			((uint8*)image->channels[1].data)[p] = pix[1];
			((uint8*)image->channels[2].data)[p] = pix[0];
			masklast = 1;
		}
		else
			masklast = 0;

		maskcount = 1;
		p++;

		for (x = 1; x < image->width; x++) {
			if (mask.at<uint8_t>(y + image->ypos, x + image->xpos)) // pixel is solid
			{
			cv::Vec3b pix = channels.at<cv::Vec3b>(y + image->ypos, x + image->xpos);
				((uint8*)image->channels[0].data)[p] = pix[2];
				((uint8*)image->channels[1].data)[p] = pix[1];
				((uint8*)image->channels[2].data)[p] = pix[0];
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

	Proftimer proftimer_extract_memcpy(&mprofiler, "extract_memcpy");
	memcpy(image->binary_mask.data,bitmap,mp<<2);
}

void extract(struct_image* image, void* bitmap) {
  Proftimer proftimer(&mprofiler, "extract");

	if (image->bpp==8) extract8(image, bitmap); else extract16(image, bitmap);
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
	Proftimer proftimer_inpaint_malloc(&mprofiler, "inpaint_malloc");
	uint8** chan_pointers=(uint8**)malloc(g_numchannels*sizeof(uint8*));
	proftimer_inpaint_malloc.stop();
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
}

void inpaint16(struct_image* image, uint32* edt) {
	int x,y;
	int c;
	uint32* edt_p=edt;
	uint32* mask_pointer=image->binary_mask.data;
	int maskcount,mask;
	uint32 dist,temp_dist;
	int copy,temp_copy;
	Proftimer proftimer_inpaint_malloc(&mprofiler, "inpaint_malloc");
	uint16** chan_pointers=(uint16**)malloc(g_numchannels*sizeof(uint16*));
	proftimer_inpaint_malloc.stop();
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
}

void inpaint(struct_image* image, uint32* edt) {
	Proftimer proftimer(&mprofiler, "inpaint");
	if (image->bpp==8) inpaint8(image,edt); else inpaint16(image,edt);
}

void tighten() {
  Proftimer proftimer(&mprofiler, "tighten");

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

cv::Rect get_visible_rect(const cv::Mat &mask)
{
	Proftimer proftimer_get_visible_rect(&mprofiler, "get_visible_rect");

	int xl = mask.cols, yl = mask.rows, xr = 0, yr = 0;
	/*for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uint8_t>(i, j))
			{
				if (xl > j)
					xl = j;
				if (yl > i)
					yl = i;
				if (xr < j)
					xr = j;
				if (yr < i)
					yr = i;
			}
		}
	}
	*/
	//left
	for (int j = 0; j < mask.cols; ++j)
	{
		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uint8_t>(i, j))
			{
				xl = j;
				j = mask.cols;
				break;
			}
		}
	}
	if (xl == mask.cols)
		die("xl == mask.cols: no visible pixels");
	//right
	for (int j = mask.cols - 1; j >= xl; --j)
	{
		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uint8_t>(i, j))
			{
				xr = j;
				j = xl - 1;
				break;
			}
		}
	}
	if (xr == 0)
		die("xr == 0: no visible pixels");
	//top
	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = xl; j <= xr; ++j)
		{
			if (mask.at<uint8_t>(i, j))
			{
				yl = i;
				i = mask.rows;
				break;
			}
		}
	}
	if (yl == mask.rows)
		die("yl == mask.rows: no visible pixels");
	//bottom
	for (int i = mask.rows - 1; i >= yl; --i)
	{
		for (int j = xl; j <= xr; ++j)
		{
			if (mask.at<uint8_t>(i, j))
			{
				yr = i;
				i = yl - 1;
				break;
			}
		}
	}
	if (yr == 0)
		die("yr == 0: no visible pixels");

	cv::Rect res;
	res.x = xl;
	res.y = yl;
	res.width = xr - xl + 1;
	res.height = yr - yl + 1;
	return res;
}

void mat2struct(int i, const std::string &filename, const cv::Mat &matimage, const cv::Mat &mask)
{
	Proftimer proftimer_mat2struct(&mprofiler, "mat2struct");

	#ifdef WIN32
		strcpy_s(g_images[i].filename, 256, filename.c_str());
	#else
		strncpy(g_images[i].filename, filename.c_str(), 256);
	#endif
	cv::Rect vis_rect = get_visible_rect(mask);
	I.bpp = 8;
	I.xpos = vis_rect.x;
	I.ypos = vis_rect.y;
	I.width = vis_rect.width;
	I.height = vis_rect.height;
	printf("vis_rect: %d, %d, %d, %d\n", vis_rect.x,vis_rect.y,vis_rect.width,vis_rect.height);

	g_workwidth = std::max(g_workwidth, (int)(I.xpos + I.width));
	g_workheight = std::max(g_workheight, (int)(I.ypos + I.height));

	void* untrimmed = (void*)malloc(I.width * I.height * sizeof(uint32));
	if (!untrimmed) die("not enough memory to process images");

	for (int c = 0; c < g_numchannels; ++c)
	{
		if (!(I.channels[c].data = (void*)malloc((I.width * I.height)<<(I.bpp>>4))))
			die("not enough memory for image channel");
	}

	extract_opencv(mask, matimage, &I, untrimmed);
	inpaint(&I, (uint32*)untrimmed);

	free(untrimmed);
}

void load_images(const std::vector<cv::Mat> &mats, const std::vector<cv::Mat> &masks) {
	Proftimer proftimer_load_images(&mprofiler, "load_images");
	g_numimages = mats.size();
	if (mats.size() != masks.size())
		die("mats.size() != masks.size()");

	g_images = (struct_image*)malloc(g_numimages*sizeof(struct_image));

	for (int i = 0; i < g_numimages; ++i) {
		I.reset();
		I.channels=(struct_channel*)malloc(g_numchannels*sizeof(struct_channel));
		for (int c = 0; c < g_numchannels; ++c) I.channels[c].f=0;
	}
	char buf[256];
	for (int i = 0; i < g_numimages; ++i) {
		//cv::Mat matimage = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
		//cv::Mat mask = cv::imread(std::string("mask_") + std::string(argv[i]), CV_LOAD_IMAGE_GRAYSCALE);
		sprintf(buf, "%d/", i);
		mat2struct(i, buf, mats[i], masks[i]);
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
