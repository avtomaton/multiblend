#define ACCURACY 5
// don't change this

#include "structs.h"
#include "globals.h"
#include "functions.h"
#include "defines.h"

#include <algorithm>
#include <emmintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void save_out_pyramid(int c, bool collapsed) {
	int l;
	int p;
	int x,y;
	char filename[1024];
	png_structp png_ptr;
	png_infop info_ptr;
	int png_height=0;
	FILE* f;

	#ifdef WIN32
		sprintf_s(filename,"out_pyramid%03d.png",c);
	#else
		sprintf(filename,"out_pyramid%03d.png",c);
	#endif

	for (l=0; l<g_levels; l++) png_height+=g_output_pyramid[l].h;

	png_ptr=png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		output(0,"WARNING: PNG create failed\n");
		return;
	}

	info_ptr=png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr,(png_infopp)NULL);
		return;
	}

	fopen_s(&f, filename, "wb");

	png_init_io(png_ptr, f);

	png_set_IHDR(png_ptr,info_ptr,g_output_pyramid[0].w,png_height,8,PNG_COLOR_TYPE_GRAY,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png_ptr, info_ptr);

	for (l=0; l<g_levels; l++) {
		p=0;
		for (y=0; y<g_output_pyramid[l].h; y++) {
			for (x=0; x<g_output_pyramid[l].w; x++) {
				if (l==g_levels-1 || collapsed) {
					if (g_workbpp==8)
						((uint8*)g_line0)[x]=std::max(0,((short*)g_output_pyramid[l].data)[p++]>>ACCURACY); //ACCURACY=&((short*)out_pyramid)[g_output_pyramid[l].offset];
					else
						((uint8*)g_line0)[x]=((int*)g_output_pyramid[l].data)[p++]>>16; //ACCURACY=&((short*)out_pyramid)[g_output_pyramid[l].offset];
				} else {
					if (g_workbpp==8)
						((uint8*)g_line0)[x]=((((short*)g_output_pyramid[l].data)[p++]<<2)+(0x7f<<ACCURACY)+(0x7f>>(8-ACCURACY)))>>ACCURACY; //ACCURACY=&((short*)out_pyramid)[g_output_pyramid[l].offset];
					else
						((uint8*)g_line0)[x]=((int*)g_output_pyramid[l].data)[p++]>>16; //ACCURACY=&((short*)out_pyramid)[g_output_pyramid[l].offset];
				}
			}

			for (; x<g_output_pyramid[0].w; x++) ((uint8*)g_line0)[x]=0x7f;
			p+=g_output_pyramid[l].pitch-g_output_pyramid[l].w;

			png_write_row(png_ptr, (uint8*)g_line0);
		}
	}

	fclose(f);
}

void hshrink(struct_level* upper, struct_level* lower) {
	int x,y;
	int tmp1,tmp2;
	size_t up=0;
	int* tmp=(int*)g_temp;
	int x_extra0=(upper->x0>>1)-lower->x0;
	int xlim=(upper->x1>>1)-lower->x0; // xpos on lower when we need to wrap last pixel
	int ushift=upper->pitch-upper->w;

	for (y=0; y<upper->h; y++) {
		x=0;
		if (g_workbpp==8) {
			tmp1=((short*)upper->data)[up++];
			tmp2=((short*)upper->data)[up++];
		} 
		else 
		{
			tmp1=((int*)upper->data)[up++];
			tmp2=((int*)upper->data)[up++];
		}
		tmp1 += tmp1<<1;
		tmp1 = tmp1+tmp2;
		while (x <= x_extra0) 
		{
			tmp[x++]=tmp1; // was +tmp2
		}

		if (g_workbpp==8) 
		{
			while (x<xlim) 
			{
				tmp1=tmp2+(((short*)upper->data)[up++]<<1);
				tmp2=((short*)upper->data)[up++];
				tmp[x++]=(tmp1+tmp2);
			}
			tmp1=((short*)upper->data)[up++];
		} 
		else 
		{
			while (x<xlim) {
				tmp1=tmp2+(((int*)upper->data)[up++]<<1);
				tmp2=((int*)upper->data)[up++];
				tmp[x++]=(tmp1+tmp2);
			}
			tmp1=((int*)upper->data)[up++];
		}

		tmp1+=tmp1<<1;
		tmp1+=tmp2;

		while (x<lower->pitch) 
		{
			tmp[x++]=tmp1;
		}

		up+=ushift;
		tmp+=lower->pitch;
	}
}

void vshrink(struct_level* upper, struct_level* lower) {
	int i;
	int x,y;
	size_t lp=0;
	int* tmp=(int*)g_temp;
	int y_extra0=(upper->y0>>1)-lower->y0;
	int ylim=(upper->y1>>1)-lower->y0; // ypos on lower when we need to duplicate last rows
	int lps;
	__m128i eight;
	__m128i sse_tmp1;
	__m128i sse_tmp2;
	__m128i* a;
	__m128i* b;
	__m128i* c;

	for (i=0; i<4; i++) ((int*)&eight)[i]=8;

// setup/top line/copies of top line
	if (g_workbpp==8) {
		for (x=0; x<lower->pitch; x++) ((short*)lower->data)[lp++]=(tmp[x]+(tmp[x]<<1)+tmp[x+lower->pitch]+8)>>4;
		for (y=1; y<=y_extra0; y++) { memcpy(&((short*)lower->data)[lp],lower->data,lower->pitch<<1); lp+=lower->pitch; }
	} 
	else {
		for (x=0; x<lower->pitch; x++) ((int*)lower->data)[lp++]=(tmp[x]+(tmp[x]<<1)+tmp[x+lower->pitch]+8)>>4;
		for (y=1; y<=y_extra0; y++) { memcpy(&((int*)lower->data)[lp],lower->data,lower->pitch<<2); lp+=lower->pitch; }
	}

// middle lines

	lps=lower->pitch>>2;

	a=(__m128i*)(tmp+lower->pitch);
	b=a+lps; // was (lower->pitch>>2);
	c=b+lps;

// move to __m128i offsets
	if (g_workbpp==8) {
		for (; y<ylim; y++) 
		{
			for (x=0; x<lps; x++) 
			{
				sse_tmp1=_mm_slli_epi32(*b++,1); // b*2
				sse_tmp2=_mm_add_epi32(*a++,*c++); // a+c
				sse_tmp1=_mm_add_epi32(sse_tmp1,sse_tmp2); // a+b*2+c
				sse_tmp1=_mm_add_epi32(sse_tmp1,eight);
				((__m128i*)g_line0)[x]=_mm_srai_epi32(sse_tmp1,4);
			}
			for (x=0; x<lower->pitch; x++) ((short*)lower->data)[lp++]=((int*)g_line0)[x];
			a=b;
			b=c;
			c+=lps;
		}
		tmp=(int*)(a-lps);
	} 
	else 
	{
		lp=lp>>2;

		for (; y<ylim; y++) {
			tmp+=lower->pitch<<1; // point to middle line of 3
			for (x=0; x<lps; x++) {
				sse_tmp1=_mm_slli_epi32(*b++,1); // b*2
				sse_tmp2=_mm_add_epi32(*a++,*c++); // a+c
				sse_tmp1=_mm_add_epi32(sse_tmp1,sse_tmp2); // a+b*2+c
				sse_tmp1=_mm_add_epi32(sse_tmp1,eight);
				((__m128i*)lower->data)[lp++]=_mm_srai_epi32(sse_tmp1,4);
			}
			// b now points to where c started, c points to the line beyond c
			a=b;
			b=c;
			c+=lps;
		}
		tmp=(int*)(a-lps);
		// move back to short/int offsets
		lp=lp<<2;
	}

// bottom line
	tmp+=lower->pitch<<1;

	if (g_workbpp==8) for (x=0; x<lower->pitch; x++) ((short*)lower->data)[lp++]=(tmp[x-lower->pitch]+tmp[x]+(tmp[x]<<1)+8)>>4;
	else            	for (x=0; x<lower->pitch; x++) ((int*)lower->data)[lp++]=(tmp[x-lower->pitch]+tmp[x]+(tmp[x]<<1)+8)>>4;
	y++;

// copies of bottom line
	if (g_workbpp==8) {
		for (; y<lower->h; y++) { memcpy(&((short*)lower->data)[lp],&((short*)lower->data)[lp-lower->pitch],lower->pitch<<1); lp+=lower->pitch; }
	} else {
		for (; y<lower->h; y++) { memcpy(&((int*)lower->data)[lp],&((int*)lower->data)[lp-lower->pitch],lower->pitch<<2); lp+=lower->pitch; }
	}
}

__inline void inflate_line_short(short *input, short *output, int w) {
	int x=0,ix=0,p,n;

	p=input[ix++];
	output[x++]=p;

	while (x<w-1) {
		n=input[ix++];
		output[x++]=(p+n+1)>>1;
		output[x++]=n;
		p=n;
	}

	if (!(w&1)) { // added for the only possible "even" case - the output pyramid  <=> (w%2==0)
		n=input[ix++];
		output[x++]=(p+n+1)>>1;
		w++;
	} 
	else 
	{
		while (x<w) { // all other cases have odd width
			output[x]=output[x-1];
			x++;
		}
	}
}

__inline void inflate_line_int(int *input, int *output, int w) {
	int x=0,ix=0,p,n;

	p=input[ix++];
	output[x++]=p;

	while (x<w-1) {
		n=input[ix++];
		output[x++]=(p+n+1)>>1;
		output[x++]=n;
		p=n;
	}

	while (x<w) {
		output[x]=output[x-1];
		x++;
	}
}

void hps(struct_level* upper, struct_level *lower) {
	int i;
	int x,y;
	int x_extra0=(upper->x0>>1)-lower->x0;
	int y_extra0=(upper->y0>>1)-lower->y0;
	int ylim=(upper->h+1)>>1;
	int sse_pitch;
	int lp;
	__m128i sse_mix;
	__m128i one;
	void* swap;

	if (g_workbpp==8) {
		sse_pitch=upper->pitch>>3;
		for (i=0; i<8; i++) ((short*)&one)[i]=1;
	} else {
		sse_pitch=upper->pitch>>2;
		for (i=0; i<4; i++) ((int*)&one)[i]=1;
	}

	lp=y_extra0*lower->pitch+x_extra0;
	__m128i* upper_p=((__m128i*)upper->data);

	if (g_workbpp==8) 
		inflate_line_short(&((short*)lower->data)[lp],(short*)g_line0,upper->pitch);
	else	            
		inflate_line_int(&((int*)lower->data)[lp],(int*)g_line0,upper->pitch);
	lp+=lower->pitch;

	if (g_workbpp==8) 
		for (x=0; x<sse_pitch; x++) 
			upper_p[x]=_mm_sub_epi16(upper_p[x],((__m128i*)g_line0)[x]);
	else            	
		for (x=0; x<sse_pitch; x++) 
			upper_p[x]=_mm_sub_epi32(upper_p[x],((__m128i*)g_line0)[x]);
	upper_p+=sse_pitch;

	for (y=1; y<ylim; y++) {
		if (g_workbpp==8) {
			inflate_line_short(&((short*)lower->data)[lp],(short*)g_line1,upper->pitch);
			for (x=0; x<sse_pitch; x++) {
				sse_mix=_mm_add_epi16(((__m128i*)g_line0)[x],((__m128i*)g_line1)[x]);
				sse_mix=_mm_add_epi16(sse_mix,one);
				sse_mix=_mm_srai_epi16(sse_mix,1);
				upper_p[x]=_mm_sub_epi16(upper_p[x],sse_mix);
			}
			lp+=lower->pitch;
		} 
		else 
		{
			inflate_line_int(&((int*)lower->data)[lp],(int*)g_line1,upper->pitch);
			for (x=0; x<sse_pitch; x++) {
				sse_mix=_mm_add_epi32(((__m128i*)g_line0)[x],((__m128i*)g_line1)[x]);
				sse_mix=_mm_add_epi32(sse_mix,one);
				sse_mix=_mm_srai_epi32(sse_mix,1);
				upper_p[x]=_mm_sub_epi32(upper_p[x],sse_mix);
			}
			lp+=lower->pitch;
		}
		upper_p+=sse_pitch;

		if (g_workbpp==8) 
			for (x=0; x<sse_pitch; x++) 
				upper_p[x]=_mm_sub_epi16(upper_p[x],((__m128i*)g_line1)[x]);
		else          		
			for (x=0; x<sse_pitch; x++) 
				upper_p[x]=_mm_sub_epi32(upper_p[x],((__m128i*)g_line1)[x]);
		upper_p+=sse_pitch;

		swap=g_line0;
		g_line0=g_line1;
		g_line1=swap;
	}
}

void shrink_hps(struct_level* upper, struct_level* lower) {
	hshrink(upper,lower);
	vshrink(upper,lower);
	//hps(upper,lower);
}

#ifdef NO_CUDA
void resizedown(const cv::Mat &umat, cv::Mat &lmat, const cv::Size &ofs)
{
	//printf("resizedown\n");
	int uh = umat.rows;
	int uw = umat.cols;
	int lw = (uw + 1) >> 1;
	int lh = (uh + 1) >> 1;

	cv::Mat tmp(uh, lw, CV_32SC3);
	cv::Vec3i eights(8);
	for (int y = 0; y < uh; ++y)
	{
		auto ptmp = tmp.ptr<cv::Vec3i>(y);
		auto pumat = umat.ptr<cv::Vec3s>(y);
		for (int x = 1; x < lw - 1; ++x)
		{
			ptmp[x] = pumat[2 * x - 1] + 2 * pumat[2 * x] + pumat[2 * x + 1];
		}
		ptmp[0] = 3 * pumat[0] + pumat[1];
		ptmp[lw - 1] = 3 * pumat[2 * (lw - 1)] + pumat[2 * (lw - 1) - 1];
	}

	for (int x = 0; x < lw; ++x)
	{
		for (int y = 1; y < lh - 1; ++y)
		{
			lmat.at<cv::Vec3s>(y + ofs.height, x + ofs.width) = (tmp.at<cv::Vec3i>(2 * y - 1, x) + 2 * tmp.at<cv::Vec3i>(2 * y, x) + tmp.at<cv::Vec3i>(2 * y + 1, x) + eights) / 16;
		}
		lmat.at<cv::Vec3s>(ofs.height, x + ofs.width) = (3 * tmp.at<cv::Vec3i>(0, x) + tmp.at<cv::Vec3i>(1, x) + eights) / 16;
		lmat.at<cv::Vec3s>(lh - 1 + ofs.height, x + ofs.width) = (3 * tmp.at<cv::Vec3i>(2 * (lh - 1), x) + tmp.at<cv::Vec3i>(2 * (lh - 1) - 1, x) + eights) / 16;
	}
}

void resizeup(const cv::Mat &lmat, cv::Mat &umat, const cv::Size &ofs = cv::Size(0,0))
{
	int uh = umat.rows;
	int uw = umat.cols;
	int lw = (uw + 1) >> 1;
	int lh = (uh + 1) >> 1;
	
	cv::Mat tmp(lh, uw, lmat.type());
	cv::Vec3s ones(1);
	for (int y = 0; y < lh; ++y)
	{
		auto ptmp = tmp.ptr<cv::Vec3s>(y);
		auto plmat = ofs.width + lmat.ptr<cv::Vec3s>(y + ofs.height);
		
		for (int x = 0; x < lw - 1; ++x)
		{
			ptmp[2 * x] = plmat[x];
			ptmp[2 * x + 1] = (plmat[x] + plmat[x + 1] + ones) / 2;
		}
		ptmp[(lw - 1) * 2] = plmat[lw - 1];
	}

	for (int x = 0; x < uw; ++x)
	{
		for (int y = 0; y < lh - 1; ++y)
		{
			umat.at<cv::Vec3s>(2 * y, x) = tmp.at<cv::Vec3s>(y, x);
			umat.at<cv::Vec3s>(2 * y + 1, x) = (tmp.at<cv::Vec3s>(y, x) + tmp.at<cv::Vec3s>(y + 1, x) + ones) / 2;
		}
		umat.at<cv::Vec3s>((lh - 1) * 2, x) = tmp.at<cv::Vec3s>(lh - 1, x);
	}
}
#else
void resizedown(const cv::cuda::GpuMat &umat, cv::cuda::GpuMat &lmat, const cv::Size &ofs)
{
	printf("resizeup\n");
	exit(1);
}
void resizeup(const cv::cuda::GpuMat &lmat, cv::cuda::GpuMat &umat, const cv::Size &ofs = cv::Size(0, 0))
{
	printf("resizeup\n");
	exit(1);
}
#endif

#ifdef NO_CUDA
void shrink_opencv(struct_level* upper, struct_level* lower, const cv::Mat &umat, cv::Mat &lmat)
{
	//printf("shrink_opencv\n");
	int x_extra0 = (upper->x0 >> 1) - lower->x0;
	int xlim = (upper->x1 >> 1) - lower->x0; // xpos on lower when we need to wrap last pixel
	int y_extra0 = (upper->y0 >> 1) - lower->y0;
	int ylim = (upper->y1 >> 1) - lower->y0; // ypos on lower when we need to duplicate last rows
	int lw = lower->w;
	int lh = lower->h;
	lmat = cv::Mat(lh, lw, CV_16SC3);
	resizedown(umat, lmat, cv::Size(x_extra0, y_extra0));

	for (int y = y_extra0; y <= ylim; ++y)
	{
		auto plmat = lmat.ptr<cv::Vec3s>(y);
		for (int x = 0; x < x_extra0; ++x)
			plmat[x] = plmat[x_extra0];
		for (int x = xlim + 1; x < lw; ++x)
			plmat[x] = plmat[xlim];
	}

	auto pborder = lmat.ptr<cv::Vec3s>(y_extra0);
	for (int y = 0; y < y_extra0; ++y)
	{
		auto plevel = lmat.ptr<cv::Vec3s>(y);
		for (int x = 0; x < lw; ++x)
			plevel[x] = pborder[x];
	}
	pborder = lmat.ptr<cv::Vec3s>(ylim);
	for (int y = ylim + 1; y < lh; ++y)
	{
		auto plevel = lmat.ptr<cv::Vec3s>(y);
		for (int x = 0; x < lw; ++x)
			plevel[x] = pborder[x];
	}
}

void hps_opencv(struct_level* upper, struct_level* lower, cv::Mat &umat, const cv::Mat &lmat)
{
	int x_extra0 = (upper->x0 >> 1) - lower->x0;
	int xlim = (upper->x1 >> 1) - lower->x0; // xpos on lower when we need to wrap last pixel
	int y_extra0 = (upper->y0 >> 1) - lower->y0;
	int ylim = (upper->y1 >> 1) - lower->y0; // ypos on lower when we need to duplicate last rows

	/*cv::Mat tmp = cv::Mat(ylim+1 - y_extra0, xlim+1 - x_extra0, CV_16SC3);
	for (int y = y_extra0; y <= ylim; ++y)
	{
		cv::Vec3s *pmat = tmp.ptr<cv::Vec3s>(y - y_extra0);
		const cv::Vec3s *plevel = lmat.ptr<cv::Vec3s>(y);

		for (int x = x_extra0; x <= xlim; ++x)
			pmat[x - x_extra0] = plevel[x];
	}*/

	cv::Mat tmp2(umat.size(), umat.type());
	resizeup(lmat, tmp2, cv::Size(x_extra0, y_extra0));
	//cv::resize(tmp, tmp2, cv::Size(umat.cols, umat.rows));
	umat -= tmp2;
}
#else
void shrink_opencv(struct_level* upper, struct_level* lower, const cv::cuda::GpuMat &umat, cv::cuda::GpuMat &lmat)
{
	printf("shrink_opencv\n");
	exit(1);
}
void hps_opencv(struct_level* upper, struct_level* lower, cv::cuda::GpuMat &umat, const cv::cuda::GpuMat &lmat)
{
	printf("hps_opencv\n");
	exit(1);
}
#endif

void copy_channel(int i, int c) {
	int x,y;
	struct_level* top=&PY(i,0);
	void* pixels;
	int a=0;
	int ip=0;
	int op=0;
	int x_extra0 = g_images[i].xpos-top->x0;
	int y_extra0 = g_images[i].ypos-top->y0;
	int y_extra1 = top->y1 - (g_images[i].ypos + g_images[i].height - 1);
	int xlim = g_images[i].width + x_extra0;
	int ipt;
	int mode;

	if (g_bgr) c = 2 -  c;

	mode = (g_workbpp==16)<<1|(g_images[i].bpp==16); // 0

	if (g_caching) {
		rewind(I.channels[c].f);
		(g_temp, (I.width*I.height)<<(I.bpp>>4), 1, I.channels[c].f);
		pixels=g_temp;
	} else {
		pixels=g_images[i].channels[c].data;
	}

	for (y=0; y<top->h-y_extra1; y++) {
		switch(mode) {
			case 0:
				a = ((uint8*)pixels)[ip++]<<ACCURACY;
				for (x = 0; x <= x_extra0; x++) 
					((short*)top->data)[op++] = a;
				ipt = ip + xlim - x;
				while (ip < ipt) 
				{
					a = ((uint8*)pixels)[ip++]<<ACCURACY;
					((short*)top->data)[op++] = a;
				}
				for (x = xlim; x < top->pitch; x++) 
					((short*)top->data)[op++] = a;
				break;
			case 1:
				a=((uint16*)pixels)[ip++]>>(8-ACCURACY);
				for (x=0; x<=x_extra0; x++) ((short*)top->data)[op++]=a;
				ipt=ip+xlim-x;
				while (ip<ipt) {
					a=((uint16*)pixels)[ip++]>>(8-ACCURACY);
					((short*)top->data)[op++]=a;
				}
				for (x=xlim; x<top->pitch; x++) ((short*)top->data)[op++]=a;
				break;
			case 2:
				a=((uint8*)pixels)[ip++];
				a=a<<16|a<<8;
				for (x=0; x<=x_extra0; x++) ((int*)top->data)[op++]=a;
				ipt=ip+xlim-x;
				while (ip<ipt) {
					a=((uint8*)pixels)[ip++];
					a=a<<16|a<<8;
					((int*)top->data)[op++]=a;
				}
				for (x=xlim; x<top->pitch; x++) ((int*)top->data)[op++]=a;
				break;
			case 3:
				a=((uint16*)pixels)[ip++]<<8;
				for (x=0; x<=x_extra0; x++) ((int*)top->data)[op++]=a;
				ipt=ip+xlim-x;
				while (ip<ipt) {
					a=((uint16*)pixels)[ip++]<<8;
					((int*)top->data)[op++]=a;
				}
				for (x=xlim; x<top->pitch; x++) ((int*)top->data)[op++]=a;
				break;
		}
		if (y==0) {
			switch(mode&2) {
				case 0:
					for (; y<=y_extra0; y++) { // not "<" , but "<=" ?
						memcpy(&((short*)top->data)[op],&((short*)top->data)[op-top->pitch],top->pitch<<1);
						op+=top->pitch;
					}
					break;
				case 2:
					for (; y<y_extra0; y++) {
						memcpy(&((int*)top->data)[op],&((int*)top->data)[op-top->pitch],top->pitch<<2);
						op+=top->pitch;
					}
					break;
			}
		}
	}
	switch(mode&2) {
		case 0:
			for (; y<top->h; y++) {
				memcpy(&((short*)top->data)[op],&((short*)top->data)[op-top->pitch],top->pitch<<1);
				op+=top->pitch;
			}
			break;
		case 2:
			for (; y<top->h; y++) {
				memcpy(&((int*)top->data)[op],&((int*)top->data)[op-top->pitch],top->pitch<<2);
				op+=top->pitch;
			}
			break;
	}

	if (!g_caching) 
	{
		free(pixels);
	}
}

#ifdef NO_CUDA
void copy_channel_opencv(int i)
{
	printf("copy_channel_opencv(%d)\n",i);
	struct_level* level = &PY(i, 0);
	int lw = level->w;
	int lh = level->h;
	g_cvmatpyramids[0] = cv::Mat(lh, lw, CV_16SC3);
	int x_extra0 = g_images[i].xpos - level->x0;
	int y_extra0 = g_images[i].ypos - level->y0;
	int y_extra1 = level->y1 - (g_images[i].ypos + g_images[i].height - 1);
	int xlim = g_images[i].width + x_extra0;
	int ylim = g_images[i].height + y_extra0;

	for (int y = y_extra0; y < ylim; ++y)
	{
		const cv::Vec3b *pmat = g_images[i].xpos - x_extra0 + g_cvmats[i].ptr<cv::Vec3b>(y - y_extra0 + g_images[i].ypos);
		cv::Vec3s *plevel = g_cvmatpyramids[0].ptr<cv::Vec3s>(y);
		
		for (int x = x_extra0; x < xlim; ++x)
		{
			plevel[x] = pmat[x];
			plevel[x][0] <<= ACCURACY;
			plevel[x][1] <<= ACCURACY;
			plevel[x][2] <<= ACCURACY;
		}

		for (int x = 0; x < x_extra0; ++x)
			plevel[x] = plevel[x_extra0];
		for (int x = xlim; x < lw; x++)
			plevel[x] = plevel[xlim - 1];
	}
	cv::Vec3s *pborder = g_cvmatpyramids[0].ptr<cv::Vec3s>(y_extra0);
	for (int y = 0; y < y_extra0; ++y)
	{
		cv::Vec3s *plevel = g_cvmatpyramids[0].ptr<cv::Vec3s>(y);
		for (int x = 0; x < lw; ++x)
			plevel[x] = pborder[x];
	}
	pborder = g_cvmatpyramids[0].ptr<cv::Vec3s>(ylim - 1);
	for (int y = ylim; y < lh; ++y)
	{
		cv::Vec3s *plevel = g_cvmatpyramids[0].ptr<cv::Vec3s>(y);
		for (int x = 0; x < lw; ++x)
			plevel[x] = pborder[x];
	}
}
#else
void copy_channel_opencv(int i)
{
	printf("copy_channel_opencv\n");
	exit(1);
}
#endif

#define NEXT_MASK { \
	pixel.f=*mask++; \
	if (pixel.i<0) { \
		count=-pixel.i; \
		pixel.f=*mask++; \
	} else count=1; \
}

void mask_into_output(struct_level* input, float* mask, struct_level* output, bool first) {
	int x,y;
	void* input_line;
	int count;
	int limcount;
	int x_extra,y_extra;
	void* out_p=((void*)output->data);
	int xlim=std::min(output->w,input->x0+input->w);
	int ylim=std::min(output->h,input->y0+input->h);
	int bpp_shift=g_workbpp>>3;
	intfloat pixel;

	if (first) memset(output->data,0,(output->pitch*output->h)<<bpp_shift);

	input_line = input->data;

	x=0;
	x_extra = input->x0;
	if (x_extra<0) {
		x_extra=0;
		x-=input->x0;
	}
	input_line=(void*)&((char*)input_line)[(x-x_extra)<<bpp_shift];

	y_extra=input->y0;
	if (y_extra<0) {
		input_line=(void*)&((char*)input_line)[(-y_extra*input->pitch)<<bpp_shift];
		y_extra=0;
	}

	// advance mask pointer to first active line
	x=output->w*y_extra;
	while (x>0) {
		NEXT_MASK;
		x-=count;
	}

	// advance output pointer to first active line
	//	out_p+=output->pitch*y_extra;
	out_p=(void*)&((char*)out_p)[(output->pitch*y_extra)<<bpp_shift];

	for (y=y_extra; y<ylim; y++) {
		// advance mask pointer to correct x position
		x=0;
		while (x<x_extra) {
			NEXT_MASK;
			x+=count;
		}

		count=x-x_extra;
		x-=count;

		// mask in active pixels
		while (x<xlim) {
			if (count==0) NEXT_MASK;
			if (pixel.f==0) 
			{
				x+=count;
				count=0;
			} 
			else if (pixel.f==1) 
			{
				limcount=xlim-x;
				if (limcount>count) limcount=count;
				if (g_workbpp==8)
					memcpy(&((short*)out_p)[x], &((short*)input_line)[x], limcount<<bpp_shift);
				else
					memcpy(&((int*)out_p)[x], &((int*)input_line)[x], limcount<<bpp_shift);
				x+=count;
				count=0;
			} 
			else 
			{
				if (g_workbpp==8)
					((short*)out_p)[x]+=(int)(((int16*)input_line)[x]*pixel.f+0.5);
				else
					((int*)out_p)[x]+=(int)(((int*)input_line)[x]*pixel.f+0.5);
				x++;
				count--;
			}
		}

		// advance mask pointer to next line
		x+=count;
		while (x<output->w) {
			NEXT_MASK;
			x+=count;
		}

		out_p=(void*)&((char*)out_p)[output->pitch<<bpp_shift];
		input_line=(void*)&((char*)input_line)[input->pitch<<bpp_shift];
	}
}

void mask_into_output_opencv(int i, int l, bool first)
{
	int chsize = g_cvmatpyramids[l].channels();

	if (first)
	{
		#ifdef NO_CUDA
		g_cvoutput_pyramid[l] = cv::Mat::zeros(g_cvmaskpyramids[i][l].size(), g_cvmatpyramids[l].type());
		#else
		g_cvoutput_pyramid[l] = cv::cuda::GpuMat(g_cvmaskpyramids[i][l].size(), g_cvmatpyramids[l].type(), cv::Scalar(0));
		#endif
	}
	int x_extra0, y_extra0;
	int xlim, ylim;

	if (l == 0)
	{
		x_extra0 = g_images[i].xpos - PY(i, l).x0;
		y_extra0 = g_images[i].ypos - PY(i, l).y0;
		xlim = g_images[i].width + x_extra0;
		ylim = g_images[i].height + y_extra0;
	}
	else
	{
		x_extra0 = (PY(i, l - 1).x0 >> 1) - PY(i, l).x0;
		xlim = (PY(i, l - 1).x1 >> 1) - PY(i, l).x0; // xpos on lower when we need to wrap last pixel
		y_extra0 = (PY(i, l - 1).y0 >> 1) - PY(i, l).y0;
		ylim = (PY(i, l - 1).y1 >> 1) - PY(i, l).y0; // ypos on lower when we need to duplicate last rows
	}
	int d;
	d = y_extra0 + PY(i, l).y0;
	if (d < 0) 
		y_extra0 -= d;
	d = x_extra0 + PY(i, l).x0;
	if (d < 0) 
		x_extra0 -= d;
	d = ylim + PY(i, l).y0;
	if (d > g_cvmaskpyramids[i][l].rows) 
		ylim = g_cvmaskpyramids[i][l].rows - PY(i, l).y0;
	d = xlim + PY(i, l).x0;
	if (d > g_cvmaskpyramids[i][l].cols)
		xlim = g_cvmaskpyramids[i][l].cols - PY(i, l).x0;

	for (int y = y_extra0; y < ylim; ++y)
	{
		auto pmat = g_cvmatpyramids[l].ptr<cv::Vec3s>(y);
		auto pout = PY(i, l).x0 + g_cvoutput_pyramid[l].ptr<cv::Vec3s>(y + PY(i, l).y0);
		auto pmask = PY(i, l).x0 + g_cvmaskpyramids[i][l].ptr<mask_t>(y + PY(i, l).y0);
		for (int x = x_extra0; x < xlim; ++x)
		{
			if (pmask[x] == 0)
				continue;
			else if (pmask[x] == max_mask_value)
				pout[x] = pmat[x];
			else
			{
				for (int c = 0; c < 3; ++c)
					pout[x][c] += (int)((float)pmask[x] * pmat[x][c] / max_mask_value + 0.5);
			}
		}
	}
}

void collapse(struct_level* lower, struct_level* upper) {
	int i;
	int x,y;
	int sse_pitch;
	int lp;
	__m128i sse_mix;
	__m128i one;
	void* swap;
	int y_extra=(lower->h*2-1)-upper->h;

	if (g_workbpp==8) {
		sse_pitch=upper->pitch>>3;
		for (i=0; i<8; i++) 
			((short*)&one)[i] = 1;
	} else {
		sse_pitch=upper->pitch>>2;
		for (i=0; i<4; i++) ((int*)&one)[i]=1;
	}

	lp=0; //y_extra0*lower->pitch+x_extra0;
	__m128i* upper_p=((__m128i*)upper->data);

	if (g_workbpp==8)	inflate_line_short(&((short*)lower->data)[lp],(short*)g_line0,upper->pitch);
	else            	inflate_line_int(&((int*)lower->data)[lp],(int*)g_line0,upper->pitch);
	lp+=lower->pitch;

	if (g_workbpp==8)	
		for (x=0; x<sse_pitch; x++) 
			upper_p[x]=_mm_add_epi16(upper_p[x],((__m128i*)g_line0)[x]);
	else            	
		for (x=0; x<sse_pitch; x++) 
			upper_p[x]=_mm_add_epi32(upper_p[x],((__m128i*)g_line0)[x]);
	upper_p+=sse_pitch;

	for (y=1; y<lower->h; y++) {
		if (g_workbpp==8) 
		{
			inflate_line_short(&((short*)lower->data)[lp],(short*)g_line1,upper->pitch);
			lp+=lower->pitch;
			for (x=0; x<sse_pitch; x++) 
			{
				sse_mix=_mm_add_epi16(((__m128i*)g_line0)[x],((__m128i*)g_line1)[x]);
				sse_mix=_mm_add_epi16(sse_mix,one);
				sse_mix=_mm_srai_epi16(sse_mix,1);
				upper_p[x]=_mm_add_epi16(upper_p[x],sse_mix);
			}
		} 
		else 
		{
			inflate_line_int(&((int*)lower->data)[lp],(int*)g_line1,upper->pitch);
			lp+=lower->pitch;
			for (x=0; x<sse_pitch; x++) 
			{
				sse_mix=_mm_add_epi32(((__m128i*)g_line0)[x],((__m128i*)g_line1)[x]);
				sse_mix=_mm_add_epi32(sse_mix,one);
				sse_mix=_mm_srai_epi32(sse_mix,1);
				upper_p[x]=_mm_add_epi32(upper_p[x],sse_mix);
			}
		}
		upper_p += sse_pitch;

		if (y==lower->h-1 && y_extra) break;

		if (g_workbpp==8) 
			for (x=0; x<sse_pitch; x++) 
				upper_p[x] = _mm_add_epi16(upper_p[x],((__m128i*)g_line1)[x]);
		else 
			for (x=0; x<sse_pitch; x++) 
				upper_p[x] = _mm_add_epi32(upper_p[x],((__m128i*)g_line1)[x]);
		upper_p += sse_pitch;

		swap = g_line0;
		g_line0 = g_line1;
		g_line1 = swap;
	}

}
#ifdef NO_CUDA
void collapse_opencv(const cv::Mat &lower, cv::Mat &upper)
#else
void collapse_opencv(const cv::cuda::GpuMat &lower, cv::cuda::GpuMat &upper)
#endif
{
	printf("collapse_opencv\n");
	#ifdef NO_CUDA
	cv::Mat tmp(upper.size(), upper.type());
	#else
	cv::cuda::GpuMat tmp(upper.size(), upper.type());
	#endif
	resizeup(lower, tmp);
	#ifdef NO_CUDA
	upper += tmp;
	#else
	cv::cuda::add(upper, tmp, upper, cv::cuda::GpuMat(), -1, cv::cuda::Stream::Null());
	#endif
}


void dither(struct_level* top, void* channel) {
	int i;
	int x,y;
	int p=0;
	int q;
	int dith_off=0;
	int dp=0;

	if (g_workbpp==8) 
	{
		if (RAND_MAX==32767) // 2 bytes
			for (i=0; i<1024; i++) g_dither[i]=rand()>>(15-ACCURACY);
		else 
			for (i=0; i<1024; i++) g_dither[i]=rand()>>(31-ACCURACY);

		for (y=0; y<g_workheight; y++) 
		{
			dith_off -= 32;
			if (dith_off < 0) 
				dith_off = 992;
			for (x=0; x<g_workwidth; x++) 
			{
				q = (((short*)top->data)[dp + x] + g_dither[dith_off + (x&31)])>>ACCURACY;
				if (q < 0) 
					q = 0; 
				else if (q > 255) 
					q = 0xff;
				((uint8*)channel)[p++] = q;
			}

			dp += top->pitch;
		}
	} 
	else 
	{
		if (RAND_MAX==32767) for (i=0; i<1024; i++) g_dither[i]=rand()>>7;
		else for (i=0; i<1024; i++) g_dither[i]=rand()>>23;

		for (y=0; y<g_workheight; y++) {
			dith_off-=32;
			if (dith_off<0) dith_off=992;
			for (x=0; x<g_workwidth; x++) {
				q=(((int*)top->data)[dp+x]+g_dither[dith_off+(x&31)])>>8;
				if (q<0) q=0; else if (q>0xffff) q=0xffff;
				((uint16*)channel)[p++]=q;
			}

			dp+=top->pitch;
		}
	}
}

#ifdef NO_CUDA
void dither_opencv(cv::Mat &top, cv::Mat &out)
{
	printf("dither_opencv\n");
	int dith_off = 0;
	std::vector<int> dither_array(1024);
	if (RAND_MAX == 32767) // 2 bytes
		for (int i = 0; i < 1024; ++i) 
			dither_array[i] = rand() >> (15 - ACCURACY);
	else
		for (int i = 0; i < 1024; ++i) 
			dither_array[i] = rand() >> (31 - ACCURACY);

	int chsize = top.channels();
	if (chsize == 3)
		out = cv::Mat(top.size(), CV_8UC3);
	else 
		die("out is not 3color");

	for (int y = 0; y < top.rows; ++y)
	{
		dith_off -= 32;
		if (dith_off < 0)
			dith_off = 992;

		cv::Vec3s *ptop = top.ptr<cv::Vec3s>(y);
		cv::Vec3b *pout = out.ptr<cv::Vec3b>(y);

		for (int x = 0; x < top.cols; ++x)
		{
			for (int c = 0; c < 3; ++c)
			{
				int q = (ptop[x][c] + dither_array[dith_off + (x & 31)]) >> ACCURACY;
				pout[x][c] = std::max(0, std::min(q, 255));
			}
		}
	}
}
#else
void dither_opencv(cv::cuda::GpuMat &top, cv::cuda::GpuMat &out)
{
	printf("dither_opencv\n");
	exit(1);
}
#endif

cv::Mat get_cvpyramid(const cv::Mat &mat)
{
	cv::Mat out(mat.size(), CV_8UC3);
	for (int y = 0; y < out.rows; ++y)
	{
		auto pout = out.ptr<cv::Vec3b>(y);
		auto pmat = mat.ptr<cv::Vec3s>(y);

		for (int x = 0; x < out.cols; ++x)
		{
			cv::Vec3s tmp = pmat[x];
			tmp[0] >>= ACCURACY;
			tmp[1] >>= ACCURACY;
			tmp[2] >>= ACCURACY;
			pout[x] = tmp;
		}
	}

	return out;
}

cv::Mat get_cvpyramid(struct_level* level)
{
	cv::Mat out(level->h, level->pitch, CV_8U);
	short* ptr = (short*)level->data;
	for (int y = 0; y < out.rows; ++y)
	{
		auto pout = out.ptr<uint8_t>(y);
		for (int x = 0; x < out.cols; ++x, ++ptr)
		{
			pout[x] = (*ptr) >> ACCURACY;
		}
	}
	return out;
}

void apply_mask(cv::Mat &out, cv::Mat &outmask)
{
	printf("apply_mask\n");
	for (int y = 0; y < out.rows; ++y)
	{
		auto pout = out.ptr<cv::Vec3b>(y);
		auto poutmask = outmask.ptr<uint8_t>(y);
		for (int x = 0; x < out.cols; ++x)
		{
			if (poutmask[x])
			{
				pout[x][0] = 0xff;
				pout[x][1] = 0xff;
				pout[x][2] = 0xff;
			}
		}
	}
}

void blend() {
	int i;
	int l;
	int c;
	int temp;
	
	size_t mem_image_max=0;
	size_t mem_temp=0;
	size_t mem_temp_max=0;
	
	my_timer timer;
	int pitch_plus;
	int size_of;

	output(1,"blending...\n");

	if (g_workbpp==8) 
		pitch_plus=7; 
	else 
		pitch_plus=3;

// dimension pyramid structs
	size_t msize = g_levels*sizeof(struct_level);
	g_output_pyramid=(struct_level*)malloc(msize);

	for (i=0; i<g_numimages; i++) {
		size_t mem_image = 0;
		g_images[i].pyramid=(struct_level*)malloc(msize);

		for (l=0; l<g_levels; l++) {
			PY(i,l).offset=mem_image;

			if (l==0) 
			{
				PY(i,l).x0=(g_images[i].xpos-1)&~1; // -1, round down to nearest even
				PY(i,l).y0=(g_images[i].ypos-1)&~1;
				PY(i,l).x1=(g_images[i].xpos+g_images[i].width+1)&~1;
				PY(i,l).y1=(g_images[i].ypos+g_images[i].height+1)&~1;
			} 
			else 
			{
				PY(i,l).x0=((PY(i,l-1).x0>>1)-1)&~1;
				PY(i,l).y0=((PY(i,l-1).y0>>1)-1)&~1;
				PY(i,l).x1=((PY(i,l-1).x1>>1)+2)&~1;
				PY(i,l).y1=((PY(i,l-1).y1>>1)+2)&~1;
			}

			PY(i,l).w=PY(i,l).x1+1-PY(i,l).x0;
			PY(i,l).h=PY(i,l).y1+1-PY(i,l).y0;
			PY(i,l).pitch=(PY(i,l).w+pitch_plus)&(~pitch_plus);

			mem_image+=PY(i,l).pitch*PY(i,l).h;
		}

		if (g_levels>1)
			mem_temp=PY(i,0).h*PY(i,1).pitch;
		else
			mem_temp=0;
		if (mem_temp>mem_temp_max) mem_temp_max=mem_temp;

		if (mem_image>mem_image_max) mem_image_max=mem_image;
	}

	size_t mem_out = 0;
	for (l=0; l<g_levels; l++) {
		g_output_pyramid[l].offset=mem_out;

		if (l==0) {
			g_output_pyramid[l].x0=0;
			g_output_pyramid[l].y0=0;
			g_output_pyramid[l].w=g_workwidth;
			g_output_pyramid[l].h=g_workheight;
		} else {
			g_output_pyramid[l].x0=0;
			g_output_pyramid[l].y0=0;
			g_output_pyramid[l].w=(g_output_pyramid[l-1].w+2)>>1;
			g_output_pyramid[l].h=(g_output_pyramid[l-1].h+2)>>1;
		}

		g_output_pyramid[l].x1=g_output_pyramid[l].w-1;
		g_output_pyramid[l].y1=g_output_pyramid[l].h-1;
		g_output_pyramid[l].pitch=(g_output_pyramid[l].w+pitch_plus)&~pitch_plus;
		
		mem_out+=g_output_pyramid[l].pitch*g_output_pyramid[l].h;
	}

	if (g_workbpp==8) 
		size_of=sizeof(short); 
	else 
		size_of=sizeof(int);

#ifdef NO_OPENCV
	void* image_pyramid=_aligned_malloc(mem_image_max*size_of,16);
	if (!image_pyramid) die("Couldn't allocate memory for image pyramid!");
	void* out_pyramid=_aligned_malloc(mem_out*size_of,16);
	if (!out_pyramid) die("Couldn't allocate memory for output pyramid!");
	mem_temp_max=std::max(mem_temp_max*sizeof(int),g_cache_bytes);
	if (mem_temp_max>0) {
		g_temp=_aligned_malloc(mem_temp_max,16); // was *sizeof(int)
		if (!g_temp) die("Couldn't allocate enough temporary memory!");
	}
	for (i=0; i<g_numimages; i++) {
		for (l=0; l<g_levels; l++) {
			if (g_workbpp==8)
				PY(i,l).data=&((short*)image_pyramid)[PY(i,l).offset];
			else
				PY(i,l).data=&((int*)image_pyramid)[PY(i,l).offset];
		}
	}

	for (l=0; l<g_levels; l++) {
		if (g_workbpp==8)
			g_output_pyramid[l].data=&((short*)out_pyramid)[g_output_pyramid[l].offset];
		else
			g_output_pyramid[l].data=&((int*)out_pyramid)[g_output_pyramid[l].offset];
	}
	g_dither=(int*)_aligned_malloc(1024<<2,16);

// iterate over channels/images, create pyramids, copy/add to output
	double copy_time=0;
	double shrink_time=0;
	double collapse_time=0;
	double dither_time=0;
	double mio_time=0;

	g_out_channels=(void**)malloc(g_numchannels*sizeof(void*));
	
	/*std::vector<std::vector<std::vector<cv::Mat> > > channels_pyramid(g_numimages);
	std::vector<std::vector<cv::Mat> > channels_outpyramid(g_levels);
	for (i = 0; i<g_numimages; i++)
	{ 
		channels_pyramid[i].resize(g_levels);
		for (l = 0; l < g_levels; l++)
			channels_pyramid[i][l].resize(g_numchannels);
	}
	for (l = 0; l < g_levels; l++)
		channels_outpyramid[l].resize(g_numchannels);
	*/

	for (c=0; c<g_numchannels; c++) {
		for (i=0; i<g_numimages; i++) {
			timer.set();
			copy_channel(i,c); // also frees channel when finished (unless input caching is on)
			copy_time+=timer.read();

			timer.set();
			for (l = 0; l < g_levels - 1; l++)
			{
				shrink_hps(&PY(i, l), &PY(i, l + 1));
				hps(&PY(i, l), &PY(i, l + 1));
				//channels_pyramid[i][l][g_numchannels - 1 - c] = get_cvpyramid(&PY(i, l));
				//if (l == g_levels - 2)
				//	channels_pyramid[i][l+1][g_numchannels - 1 - c] = get_cvpyramid(&PY(i, l+1));
			}

			shrink_time+=timer.read();

			timer.set();
			for (l = 0; l < g_levels; l++)
			{
				mask_into_output(&PY(i, l), g_images[i].masks[l], &g_output_pyramid[l], i == 0);
				//channels_outpyramid[l][g_numchannels - 1 - c] = get_cvpyramid(&g_output_pyramid[l]);
			}
			mio_time+=timer.read();
		}

		if (g_save_out_pyramids) save_out_pyramid(c,false);

		timer.set();
		for (l=g_levels-1; l>0; l--) collapse(&g_output_pyramid[l],&g_output_pyramid[l-1]);
		collapse_time+=timer.read();

		if (i==g_numimages-1 && c==g_numchannels-1) {
			_aligned_free(image_pyramid);
			_aligned_free(g_temp);
		}

		temp=(g_workwidth*g_workheight)<<(g_workbpp>>4);
		g_out_channels[c]=_aligned_malloc(temp,0);
		if (!g_out_channels[c]) die("not enough memory for output channel!");

		timer.set();
		dither(&g_output_pyramid[0], g_out_channels[c]);
		dither_time+=timer.read();
	}

	/*
	for (i = 0; i < g_numimages; i++)
	{
		for (l = 0; l < g_levels; l++)
		{
			cv::Mat m;
			std::string outstring = "pyramid\\pyramid";
			outstring += "_";
			outstring += std::to_string(i);
			outstring += "_";
			outstring += std::to_string(l);
			outstring += ".png";
			cv::merge(channels_pyramid[i][l], m);
			cv::imwrite(outstring, m);
		}
	}
	for (l = 0; l < g_levels; l++)
	{
		cv::Mat m;
		std::string outstring = "outpyramid\\outpyramid";
		outstring += "_";
		outstring += std::to_string(l);
		outstring += ".png";
		cv::merge(channels_outpyramid[l], m);
		cv::imwrite(outstring, m);
	}
	*/

	if (g_timing) {
		printf("\n");
		report_time("copy", copy_time);
		report_time("shrink", shrink_time);
		report_time("merge", mio_time);
		report_time("collapse", collapse_time);
		report_time("dither", dither_time);
		printf("\n");
	}

	_aligned_free(g_dither);
	_aligned_free(out_pyramid);

#else
	///////////////////////////////////////////////////////////////////////////////////////////
	
	g_cvmatpyramids.resize(g_levels);
	g_cvoutput_pyramid.resize(g_levels);

	for (i = 0; i<g_numimages; i++) {
		copy_channel_opencv(i);

		for (l = 0; l < g_levels - 1; l++)
		{
			shrink_opencv(&PY(i, l), &PY(i, l + 1), g_cvmatpyramids[l], g_cvmatpyramids[l + 1]);
			hps_opencv(&PY(i, l), &PY(i, l + 1), g_cvmatpyramids[l], g_cvmatpyramids[l + 1]);
			//channels_pyramid[i][l][0] = get_cvpyramid(g_cvmatpyramids[l]);
			//if (l == g_levels - 2)
			//	channels_pyramid[i][l + 1][0] = get_cvpyramid(g_cvmatpyramids[l+1]);
		}

		for (l = 0; l < g_levels; l++)
		{
			mask_into_output_opencv(i, l, i == 0);
			//channels_outpyramid[l][0] = get_cvpyramid(g_cvoutput_pyramid[l]);
		}
	}

	/*printf("print pyramid_opencv\n");
	for (i = 0; i < g_numimages; i++)
	{
		for (l = 0; l < g_levels; l++)
		{
			cv::Mat m;
			std::string outstring = "pyramid\\pyramid";
			outstring += "_";
			outstring += std::to_string(i);
			outstring += "_";
			outstring += std::to_string(l);
			outstring += "_opencv.png";
			cv::imwrite(outstring, channels_pyramid[i][l][0]);
		}
	}
	printf("print outpyramid_opencv\n");
	for (l = 0; l < g_levels; l++)
	{
		cv::Mat m;
		std::string outstring = "outpyramid\\outpyramid";
		outstring += "_";
		outstring += std::to_string(l);
		outstring += "_opencv.png";
		cv::imwrite(outstring, channels_outpyramid[l][0]);
	}*/

	for (l = g_levels - 1; l > 0; l--)
		collapse_opencv(g_cvoutput_pyramid[l], g_cvoutput_pyramid[l - 1]);
	#ifdef NO_CUDA
	cv::Mat tmpout;
	#else
	cv::cuda::GpuMat tmpout;
	#endif
	/*cv::Mat tmpout2;
	tmpout = g_cvoutput_pyramid[0].clone();
	tmpout /= 1 << ACCURACY;
	tmpout.convertTo(tmpout2, CV_8U);
	std::string outstring = "collapse";
	outstring += "_";
	outstring += std::to_string(0);
	outstring += ".png";
	cv::imwrite(outstring, tmpout2);
	*/

	dither_opencv(g_cvoutput_pyramid[0], tmpout);
	#ifdef NO_CUDA
	cv::Mat outroi(tmpout, cv::Rect(0, 0, g_workwidth, g_workheight));
	g_cvout = outroi;
	#else
	cv::cuda::GpuMat outroi(tmpout, cv::Rect(0, 0, g_workwidth, g_workheight));
	outroi.download(g_cvout);
	#endif
	
	//apply_mask(g_cvout, g_cvoutmask);
	//cv::imwrite("output_opencv.png", g_cvout);
	
#endif
	///////////////////////////////////////////////////////////////////////////////////////////

	free(g_output_pyramid);
	for (i=0; i<g_numimages; i++) 
		free(g_images[i].pyramid);
}
