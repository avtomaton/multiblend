#include "structs.h"
#include "globals.h"
#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void png_mask(int i) {
	int x,y;
	int w;
	int h;
	int j;
	int l;
	int png_height=0;
	float* input;
	float m;
	intfloat ipixel;
	png_structp png_ptr;
	png_infop info_ptr;
	char filename[256];
	FILE* f;

#ifdef WIN32
	sprintf_s(filename,"mb_mask%03d.png",i);
#else
	sprintf(filename,"mb_mask%03d.png",i);
#endif

	h=g_workheight;
	for (l=0; l<g_levels; l++) {
		png_height+=h;
		h=(h+2)>>1;
	}

	fopen_s(&f,filename, "wb");
	if (!f) {
		output(0,"WARNING: couldn't save mask\n");
		return;
	}

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

	png_init_io(png_ptr, f);

	png_set_IHDR(png_ptr,info_ptr,g_workwidth,png_height,8,PNG_COLOR_TYPE_GRAY,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png_ptr, info_ptr);

	w=g_workwidth;
	h=g_workheight;
	for (l=0; l<g_levels; l++) {
		input=g_images[i].masks[l];
		for (y=0; y<h; y++) {
			x=0;
			while (x<w) {
				ipixel.f=*input++;
				if (ipixel.i<0) { // RLE
					m=*input++;
					for (j=0; j<-ipixel.i; j++) ((uint8*)g_line0)[x++]=(int)(m*255+0.5);
				} else {
					((uint8*)g_line0)[x++]=(int)(ipixel.f*255+0.5);
				}
			}

			if (y==0) while (x<g_workwidth) ((uint8*)g_line0)[x++]=128;

			png_write_row(png_ptr, (uint8*)g_line0);
		}
		w=(w+2)>>1;
		h=(h+2)>>1;
	}

	fclose(f);
}

int squish_line(float* input, float *output, int inwidth, int outwidth) {
	int p=0;
	int outp=0;
	float runningtotal=0;
	intfloat pixel;
	float outpixel;
	int readsofar=0;
	int readmore=3;
	int count;
	float lastrle=-1;

// first pixel;
	pixel.f=input[p++];
	if (pixel.i<0) { // rle sequence
		count=-pixel.i;
		pixel.f=input[p++];
		readsofar+=count;

		if (readsofar==inwidth) { // was this the only pixel? (special case)
			*(int32*)&output[outp++]=-outwidth;
			output[outp++]=pixel.f*4; // because output will be squashed with 3 other lines later
			return p;
		}

		if (count>1) {
			*(int32*)&output[outp++]=-(count>>1);
			lastrle=pixel.f*4;
			output[outp++]=lastrle;
		}

		if (count&1) {
			runningtotal=pixel.f*3;
			readmore=1;
		} else {
			runningtotal=pixel.f;
			readmore=2;
		}
	} else { // non-rle sequence
		readsofar=1;
		runningtotal=pixel.f*3;
		readmore=1;
	}

// remaining pixels
	while (readmore>0) {
		pixel.f=input[p++];

		if (readmore==2) {
			if (pixel.i<0) { // rle sequence
				count=-pixel.i;
				pixel.f=input[p++];
				readsofar+=count;
				if (readsofar==inwidth) break;
				if (count>1) {
					outpixel=runningtotal+pixel.f*3;
					if (lastrle==outpixel) {
						*(int32*)&output[outp-2]-=1;
					} else {
						output[outp++]=outpixel;
						lastrle=-1;
					}

					if (count>3) {
						if (lastrle==pixel.f) {
							*(int32*)&output[outp-2]-=(count-2)>>1;
						} else {
							*(int32*)&output[outp++]=-((count-2)>>1);
							output[outp++]=pixel.f*4;
							lastrle=pixel.f*4;
						}
					}

					if (count&1) {
						runningtotal=pixel.f*3;
						readmore=1;
					} else {
						runningtotal=pixel.f;
					}
				} else { // non-rle
					runningtotal+=pixel.f*2;
					readmore=1;
				}
			} else {
				readsofar++;
				count=1;
				if (readsofar==inwidth) break;
				runningtotal+=pixel.f*2;
				readmore=1;
			}
		} else { // readmore==1
			if (pixel.i<0) { // rle
				count=-pixel.i;
				pixel.f=input[p++];
				readsofar+=count;

				if (readsofar==inwidth) break;

				outpixel=runningtotal+pixel.f;
				if (lastrle==outpixel) {
					*(int32*)&output[outp-2]-=1;
				} else {
					output[outp++]=outpixel;
					lastrle=-1;
				}

				if (count>2) {
					if (lastrle==pixel.f) {
						*(int32*)&output[outp-2]-=(count-1)>>1;
					} else {
						*(int32*)&output[outp++]=-((count-1)>>1);
						output[outp++]=pixel.f*4;
						lastrle=pixel.f*4;
					}
				}

				if (count&1) {
					runningtotal=pixel.f;
					readmore=2;
				} else {
					runningtotal=pixel.f*3; // was +=pixel*2
				}
			} else { // non-rle
				readsofar++;
				count=1;

				if (readsofar==inwidth) break;

				outpixel=runningtotal+pixel.f;

				if (lastrle==outpixel) {
					*(int32*)&output[outp-2]-=1;
				} else {
					output[outp++]=outpixel; // was pixel
					lastrle=-1;
				}

				runningtotal=pixel.f;
				readmore=2;
			}
		}
	} // endwhile

	if (readmore==1) {
		runningtotal+=pixel.f;
	} else {
		runningtotal+=pixel.f*3;
	}

	outpixel=runningtotal;

	if (outpixel==lastrle) {
		*(int*)&output[outp-2]-=1;
	} else {
		output[outp++]=outpixel;
		lastrle=-1;
	}

	count=(count+1-(inwidth&1))>>1;
	if (count>0) {
		if (count>1) {
			*(int*)&output[outp++]=-count;
		}
		output[outp++]=pixel.f*4;
	}

	return p;
}

int squash_lines(float* a, float* b, float* c, float* o, int width) {
	int i;
	float* pointer[3];
	int count[3];
	intfloat pixel[3];
	int p=0;
	int x=0;
	int mincount;

	pointer[0]=a; pointer[1]=b; pointer[2]=c;
	count[0]=0; count[1]=0; count[2]=0;

	while (x<width) {
		mincount=0x7fffffff;
		for (i=0; i<3; i++) {
			if (count[i]==0) {
				pixel[i].f=*pointer[i]++;
				if (pixel[i].i<0) {
					count[i]=-pixel[i].i;
					pixel[i].f=*pointer[i]++;
				} else {
					count[i]=1;
				}
			}
			if (count[i]<mincount) mincount=count[i];
		}

		if (mincount >= 1) 
			*(int*)&o[p++]=-mincount;

		o[p++]=(float)((pixel[0].f+pixel[1].f*2+pixel[2].f)*0.0625);

		for (i=0; i<3; i++) {
			count[i]-=mincount;
		}

		x+=mincount;
	}

	return p;
}

void shrink_mask(float* input, float **output_pointer, int inwidth, int inheight, int outwidth, int outheight) {
	int input_p=0;
	int output_p=0;
	int lines_read;
	int size=g_workwidth<<1;
	int c,y;
	void* swap;
	float* output=(float*)malloc(size*sizeof(float));

	input_p+=squish_line(&input[input_p],(float*)g_line0,inwidth,outwidth);
	input_p+=squish_line(&input[input_p],(float*)g_line2,inwidth,outwidth);
	output_p+=squash_lines((float*)g_line0,(float*)g_line0,(float*)g_line2,&output[output_p],outwidth);
	lines_read=2;

	for (y=1; y<outheight; y++) {
		swap=g_line0;
		g_line0=g_line2;
		g_line2=swap;

		c=squish_line(&input[input_p],(float*)g_line1,inwidth,outwidth);
		lines_read++;
		if (lines_read<inheight) input_p+=c;

		c=squish_line(&input[input_p],(float*)g_line2,inwidth,outwidth);
		lines_read++;
		if (lines_read<inheight) input_p+=c;

		output_p+=squash_lines((float*)g_line0,(float*)g_line1,(float*)g_line2,&output[output_p],outwidth);

		if (output_p+(g_workwidth)>size) {
			size=size<<1;
			output=(float*)realloc(output,size*sizeof(float));
		}
	}

	output=(float*)realloc(output,output_p*sizeof(float));
	*output_pointer=output;
}

void extract_top_masks() {
	int i;
	int x,y;
	int* p=(int*)malloc(g_numimages*sizeof(int));
	int* size=(int*)malloc(g_numimages*sizeof(int));
	int seam_p=0;
	uint32 seam;
	int32 this_i,this_count,last_i;

	for (i=0; i<g_numimages; i++) {
		p[i]=0;
		size[i]=g_workwidth<<1;
		g_images[i].masks[0]=(float*)malloc(size[i]*sizeof(float));
	}

	for (y=0; y<g_workheight; y++) {
		x=g_workwidth;
		last_i=-1;
		while (x>0) {
			seam=g_seams[seam_p++];
			this_i=seam&0xff;
			this_count=seam>>8;
			x-=this_count;

			if (last_i==-1) {
				for (i=0; i<g_numimages; i++) {
					((int32*)g_images[i].masks[0])[p[i]++]=-this_count;
					if (i==this_i) g_images[i].masks[0][p[i]++]=1; else g_images[i].masks[0][p[i]++]=0;
				}
			} else {
				for (i=0; i<g_numimages; i++) {
					if (i==last_i) {
						((int32*)g_images[i].masks[0])[p[i]++]=-this_count;
						g_images[i].masks[0][p[i]++]=0;
					} else if (i==this_i) {
						((int32*)g_images[i].masks[0])[p[i]++]=-this_count;
						g_images[i].masks[0][p[i]++]=1;
					} else {
						((int32*)g_images[i].masks[0])[p[i]-2]-=this_count;
					}
				}
			}

			last_i=this_i;
		}

		for (i=0; i<g_numimages; i++) {
			if (p[i]+(g_workwidth)>size[i]) {
				size[i]+=g_workwidth<<1;
				g_images[i].masks[0]=(float*)realloc(g_images[i].masks[0],size[i]*sizeof(float));
			}
		}
	}

	for (i=0; i<g_numimages; i++) {
		size[i]=p[i];
		g_images[i].masks[0]=(float*)realloc(g_images[i].masks[0],size[i]*sizeof(float));
	}

	free(p);
	free(size);
}

cv::Mat top_mask_to_cvmat(int i, int l, int w, int h)
{
	printf("top_mask_to_cvmat(%d, %d, %d, %d)\n", i, l, w, h);
	cv::Mat out(h, w, CV_8U);
	float* pmask = g_images[i].masks[l];
	int p = 0;
	intfloat pix;

	int index = 0;
	int y = 0;
	int x = 0;
	uint8_t *pout = out.ptr<uint8_t>(0);
	while (index < h*w)
	{
		pix.f = pmask[p++];
		int count = -pix.i;
		float num = pmask[p++];
		for (int k = 0; k < count; ++k)
		{
			pout[x++] = std::max<int>(0, std::min<int>(num * 128,255));
			if (x == w)
			{
				pout = out.ptr<uint8_t>(++y);
				x = 0;
			}
			++index;
		}
	}

	return out;
}

void write_mask(cv::Mat &mask, int i, int l)
{
	//cv::Mat mat = top_mask_to_cvmat(i, l, ow, oh);
	mask *= 64;
	cv::Mat tmpmask;
	mask.convertTo(tmpmask, CV_8U);
	std::string out_path = std::string("maskpyramids\\maskpyramid_") + std::to_string(i) + std::string("_") + std::to_string(l) + std::string(".bmp");
	cv::imwrite(out_path, tmpmask);
	mask /= 64.0;
}

void shrink_masks() {
	int i,l;
	int w,h;
	int ow,oh;

	for (i=0; i<g_numimages; i++) {
		w=g_workwidth;
		h=g_workheight;
		for (l=0; l<g_levels-1; l++) {
			ow=(w+2)>>1;
			oh=(h+2)>>1;
			shrink_mask(g_images[i].masks[l], &g_images[i].masks[l+1], w, h, ow, oh);
			w=ow;
			h=oh;
		}
	}

	if (g_savemasks) {
		output(1,"saving masks...\n");
		for (i=0; i<g_numimages; i++) png_mask(i);
	}
}

void shrink_masks_opencv() {
	printf("shrink_masks_opencv\n");

	int i, l;
	int w, h;
	int ow, oh;

	for (i = 0; i<g_numimages; i++) {
		w = g_cvmaskpyramids[i][0].cols;
		h = g_cvmaskpyramids[i][0].rows;
		for (l = 0; l < g_levels - 1; ++l) {
			ow = (w + 1) >> 1;
			oh = (h + 1) >> 1;
			if (ow % 2 != 0 && oh % 2 != 0)
			{
				cv::resize(g_cvmaskpyramids[i][l], g_cvmaskpyramids[i][l + 1], cv::Size(ow, oh));
			}
			else
			{
				cv::Mat tmp;
				cv::resize(g_cvmaskpyramids[i][l], tmp, cv::Size(ow, oh));
				
				if (ow % 2 == 0) ++ow;
				if (oh % 2 == 0) ++oh;
				g_cvmaskpyramids[i][l + 1] = cv::Mat(oh, ow, g_cvmaskpyramids[i][l].type());
				for (int y = 0; y < tmp.rows; ++y)
				{
					const float* ptmp = tmp.ptr<float>(y);
					float *pmat = g_cvmaskpyramids[i][l + 1].ptr<float>(y);
					for (int x = 0; x < tmp.cols; ++x)
						pmat[x] = ptmp[x];

					for (int x = tmp.cols; x < ow; ++x)
						pmat[x] = pmat[tmp.cols - 1];
				}

				float *pborder = g_cvmaskpyramids[i][l + 1].ptr<float>(tmp.rows - 1);
				for (int y = tmp.rows; y < oh; ++y)
				{
					float *pmat = g_cvmaskpyramids[i][l + 1].ptr<float>(y);
					for (int x = 0; x < ow; ++x)
						pmat[x] = pborder[x];
				}
			}
			
			w = ow;
			h = oh;
		}
	}
}

void extract_top_masks_opencv()
{
	printf("extract_top_masks_opencv\n");
	g_cvmaskpyramids.resize(g_numimages);
	int cols = g_workwidth;
	int rows = g_workheight;

	if (cols % 2 == 0) ++cols;
	if (rows % 2 == 0) ++rows;

	for (int i = 0; i < g_numimages; ++i)
	{
		g_cvmaskpyramids[i].resize(g_levels);
		g_cvmaskpyramids[i][0] = cv::Mat::zeros(rows, cols, CV_32F);
	}
	std::vector<float*> pmasks(g_numimages);
	for (int y = 0; y < g_workheight; ++y)
	{
		const uint8_t *pseam = g_cvseams.ptr(y);
		for (int i = 0; i < g_numimages; ++i)
			pmasks[i] = g_cvmaskpyramids[i][0].ptr<float>(y);
		for (int x = 0; x < g_workwidth; ++x)
		{
			if (pseam[x] < 0 || pseam[x] >= g_numimages)
				continue;
			pmasks[pseam[x]][x] = 1.0f;
		}

		for (int x = g_workwidth; x < cols; ++x)
			for (int i = 0; i < g_numimages; ++i)
				pmasks[i][x] = pmasks[i][g_workwidth - 1];
	}

	for (int i = 0; i < g_numimages; ++i)
	{
		const float *pborder = g_cvmaskpyramids[i][0].ptr<float>(g_workheight - 1);
		for (int y = g_workheight; y < rows; ++y)
		{
			pmasks[i] = g_cvmaskpyramids[i][0].ptr<float>(y);
			for (int x = 0; x < cols; ++x)
				pmasks[i][x] = pborder[x];
		}
	}
}

cv::Mat get_mask(int i, int l, int w, int h)
{
	int32 this_count = 0;
	float mask;
	int p = 0;
	int32 counter = 0;

	cv::Mat out(h, w, CV_32F);
	for (int y = 0; y < h; ++y)
	{
		float *pout = out.ptr<float>(y);
		for (int x = 0; x < w; ++x)
		{
			if (counter >= this_count)
			{
				counter = 0;
				this_count = -((int32*)g_images[i].masks[l])[p++];
				mask = g_images[i].masks[l][p++];
			}

			pout[x] = mask;
			counter++;
		}
	}
	return out;
}

void mask_pyramids() {
	Proftimer proftimer(&mprofiler, "mask_pyramids");
	output(1,"masks...\n");

	extract_top_masks();
	extract_top_masks_opencv();
	shrink_masks();
	shrink_masks_opencv();

	/*for (int i = 0; i < g_numimages; ++i)
	{
		int w = g_workwidth;
		int h = g_workheight;
		for (int l = 0; l < g_levels; ++l)
		{
			cv::Mat mat = get_mask(i, l, w, h);
			cv::Mat tmp;
			mat.convertTo(tmp, CV_8U);
			std::string outstring = "maskpyramids\\maskpyramid";
			outstring += "_";
			outstring += std::to_string(i);
			outstring += "_";
			outstring += std::to_string(l);
			outstring += ".png";
			cv::imwrite(outstring, tmp * 255);
			w = (w + 2) >> 1;
			h = (h + 2) >> 1;
		}
	}

	for (int i = 0; i < g_numimages; ++i)
	{
		for (int l = 0; l < g_levels; ++l)
		{
			cv::Mat mat = g_cvmaskpyramids[i][l];
			cv::Mat tmp;
			mat.convertTo(tmp, CV_8U);
			std::string outstring = "maskpyramids\\maskpyramid";
			outstring += "_";
			outstring += std::to_string(i);
			outstring += "_";
			outstring += std::to_string(l);
			outstring += "_opencv.png";
			cv::imwrite(outstring, tmp * 255);
		}
	}
	*/
}
