#include "globals.h"
#include "functions.h"
#include "defines.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void seam_png(int mode, const char* filename) {
	int x;
	int y;
	int count,i;
	int* maskcount=(int*)malloc(g_numimages*sizeof(int));
	int* masklimit=(int*)malloc(g_numimages*sizeof(int));
	int* mask=(int*)malloc(0x100*sizeof(int));
	int mincount;
	int xorcount;
	int xorimage;
	int stop;
	uint32 temp;
	uint32* seam_p;
	png_structp png_ptr;
	png_infop info_ptr;
	double base=2;
	double rad;
	double r,g,b;
	FILE* f;

	if (!g_palette) {
		g_palette=(png_color*)malloc(256*sizeof(png_color));

		for (i=0; i<255; i++) {
			rad=base;
			r=std::max<double>(0,std::min<double>(1.0,std::min<double>(rad,4-rad)));
			rad+=2; if (rad>=6) rad-=6;
			g=std::max<double>(0,std::min<double>(1.0,std::min<double>(rad,4-rad)));
			rad+=2; if (rad>=6) rad-=6;
			b=std::max<double>(0,std::min<double>(1.0,std::min<double>(rad,4-rad)));
			base+=6*0.618033988749895;
			if (base>=6) base-=6;
			g_palette[i].red=(int)(r*255+0.5);
			g_palette[i].green=(int)(g*255+0.5);
			g_palette[i].blue=(int)(b*255+0.5);
		}
		g_palette[i].red=0;
		g_palette[i].green=0;
		g_palette[i].blue=0;
	}

	switch (mode) {
		case 0 : output(1,"saving xor map...\n"); break;
		case 1 : output(1,"saving seams...\n"); break;
	}

	fopen_s(&f, filename, "wb");
	if (!f) {
		output(0,"WARNING: couldn't save seam file\n");
		free(mask);
		free(masklimit);
		free(maskcount);
		return;
	}

	png_ptr=png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		output(0,"WARNING: PNG create failed\n");
		free(mask);
		free(masklimit);
		free(maskcount);
		return;
	}

	info_ptr=png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr,(png_infopp)NULL);
		free(mask);
		free(masklimit);
		free(maskcount);
		return;
	}

	png_init_io(png_ptr, f);

	png_set_IHDR(png_ptr,info_ptr,g_workwidth,g_workheight,8,PNG_COLOR_TYPE_PALETTE,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);
	png_set_PLTE(png_ptr,info_ptr,g_palette,256);

	png_write_info(png_ptr, info_ptr);

	if (mode==0) {
		for (y=0; y<g_workheight; y++) {
			for (i=0; i<g_numimages; i++) {
				mask[i]=MASKOFF;
				if (y>=g_images[i].ypos && y<g_images[i].ypos+g_images[i].height) {
					maskcount[i]=g_images[i].xpos;
					masklimit[i]=g_images[i].xpos+g_images[i].width;
					g_images[i].binary_mask.pointer=&g_images[i].binary_mask.data[g_images[i].binary_mask.rows[y-g_images[i].ypos]];
				} else {
					maskcount[i]=g_workwidth;
					masklimit[i]=g_workwidth;
				}
			}

			x=0;
			while (x<g_workwidth) {
				mincount=g_workwidth-x;
				xorcount=0;
				for (i=0; i<g_numimages; i++) {
					if (maskcount[i]==0) {
						if (x<masklimit[i]) {
							NEXTiMASK(i);
						} else {
							mask[i]=MASKOFF;
							maskcount[i]=mincount;
						}
					}

					if (maskcount[i]<mincount) mincount=maskcount[i];
					if (mask[i]!=MASKOFF) {
						xorcount++;
						xorimage=i;
					}
				}

				stop=x+mincount;

				if (xorcount!=1) xorimage=255;

				while (x<stop) {
					((uint8*)g_line0)[x++]=xorimage;
				}

				for (i=0; i<g_numimages; i++) maskcount[i]-=mincount;
			}

			png_write_row(png_ptr, (uint8*)g_line0);
		}
	}

	else if (mode==1) {
		seam_p=g_seams;

		for (y=0; y<g_workheight; y++) {
			x=0;
			while (x<g_workwidth) {
				i=*seam_p&0xff;
				count=*seam_p++>>8;
				memset(&((uint8*)g_line0)[x],i,count);
				x+=count;
			}

			png_write_row(png_ptr, (uint8*)g_line0);
		}
	}

	fclose(f);
	free(mask);
	free(masklimit);
	free(maskcount);
}

void load_seams() {
	int x,y;
	int pd,pc;
	int size;
	int a,b;
	int count=1;
	int p=0;
	png_uint_32 pw,ph;
	uint8 sig[8];
	png_structp png_ptr;
	png_infop info_ptr;
	FILE* f;

	fopen_s(&f,g_seamload_filename,"rb");
	if (!f) die("Couldn't open seam file!");

	fread(sig, 1, 8, f);
	if (!png_check_sig(sig,8)) die("Bad PNG signature!");

	png_ptr=png_create_read_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
	if (!png_ptr) die("PNG problem");
	info_ptr=png_create_info_struct(png_ptr);
	if (!info_ptr) die("PNG problem");

	png_init_io(png_ptr,f);
	png_set_sig_bytes(png_ptr,8);
	png_read_info(png_ptr,info_ptr);
	png_get_IHDR(png_ptr, info_ptr, &pw, &ph, &pd, &pc,NULL,NULL,NULL);

	if (pw!=g_workwidth || ph!=g_workheight) die("PNG dimensions don't match workspace!");
	if (pd!=8 || pc!=PNG_COLOR_TYPE_PALETTE) die("Incorrect seam PNG format!");

	size=(g_numimages*g_workwidth)<<2;
	g_seams=(uint32*)malloc(size*sizeof(uint32));

	for (y=0; y<g_workheight; y++) {
		png_read_row(png_ptr,(png_bytep)g_line0,NULL);
		a=((uint8*)g_line0)[0]&0xff;
		x=1;
		while (x<g_workwidth) {
			b=((uint8*)g_line0)[x++]&0xff;
			if (b!=a) {
				g_seams[p++]=count<<8|a;
				g_images[a].seampresent=true;
				count=1;
				a=b;
			} else {
				count++;
			}
		}
		g_seams[p++]=count<<8|a;
		count=1;

		if ((p+((g_numimages*g_workwidth)<<1))>size) {
			size+=(g_numimages*g_workwidth)<<2;
			g_seams=(uint32*)realloc(g_seams,size*sizeof(uint32));
		}
	}

	g_seams=(uint32*)realloc(g_seams,p*sizeof(uint32));
}

#define EDT_MAX 0xfffffbff 
//11111111 11111111 11111100 11111111
#define VALMASKED(x) (x|mask[x&0xff])

void rightdownxy() 
{
	int i;
	int x;
	int y;
	int xorcount;
	int mincount=0;
	int stop;
	uint32 temp;
	int* maskcount=(int*)malloc(g_numimages*sizeof(int));
	int* masklimit=(int*)malloc(g_numimages*sizeof(int));
	int* mask=(int*)malloc(0x100*sizeof(int));
	bool lastpixel=false;
	uint32* line;
	uint32 bestval,testval;
	uint32 a,b,c,d;

	mask[255]=MASKOFF;

	y=0;
	while (y<g_workheight) {
		line=&g_edt[y*g_workwidth];

		for (i=0; i<g_numimages; i++) {
			mask[i]=MASKOFF;
			if (y>=g_images[i].ypos && y<g_images[i].ypos+g_images[i].height) {
				maskcount[i]=g_images[i].xpos;
				masklimit[i]=g_images[i].xpos+g_images[i].width;
				g_images[i].binary_mask.pointer=&g_images[i].binary_mask.data[g_images[i].binary_mask.rows[y-g_images[i].ypos]];
			} else {
				maskcount[i]=g_workwidth;
				masklimit[i]=g_workwidth;
			}
		}

		x=0;
		while (x<g_workwidth) {
			mincount=g_workwidth-x;
			xorcount=0;
			for (i=0; i<g_numimages; i++) {
				if (maskcount[i]==0) {
					if (x<masklimit[i]) {
						NEXTiMASK(i);
					} else {
						mask[i]=MASKOFF;
						maskcount[i]=mincount;
					}
				}

				if (maskcount[i]<mincount) mincount=maskcount[i];
				if (mask[i]!=MASKOFF) {
					xorcount++;
					//xorimage=i;
				}
			}

			stop=x+mincount;

			if (xorcount==1) x=stop;
			else {
				// if we're on the top line:
				if (y==0) 
				{
					if (x==0) x=1;
					while (x<stop) {
						bestval=line[x];

						testval=VALMASKED(line[x-1])+(3<<8); // changed to -1
						if (testval<bestval) bestval=testval;

						if (bestval&MASKOFF && xorcount!=0) {
							for (i=0; i<g_numimages; i++) {
								if (mask[i]==MASKON) {
									g_seamwarning=true;
									bestval=MASKOFF|i;
									if (!g_reverse) break;
								}
							}
						}

						line[x++]=bestval;
					}
				} 
				else 
				{
					// if we're not on the top line
					if (x==0) {
						testval=VALMASKED(line[-g_workwidth])+(3<<8);
						bestval=std::min(line[x],testval);

						testval=VALMASKED(line[-g_workwidth+1])+(4<<8);
						if (testval<bestval) bestval=testval;

						if (bestval&MASKOFF && xorcount!=0) {
							for (i=0; i<g_numimages; i++) {
								if (mask[i]==MASKON) {
									g_seamwarning=true;
									bestval=MASKOFF|i;
									if (!g_reverse) break;
								}
							}
						}

						line[x++]=bestval;
					}

					if (stop==g_workwidth) {
						stop=g_workwidth-1;
						lastpixel=true;
					}

					/* abc
					 dx  */
					if (x<stop) {
						a=VALMASKED(line[-g_workwidth+x-1])+(4<<8);
						b=VALMASKED(line[-g_workwidth+x])+(3<<8);
						d=VALMASKED(line[x-1])+(3<<8);

						while (x<stop) { // main bit
							temp=line[-g_workwidth+x+1];
							c=VALMASKED(temp)+(4<<8);

							bestval=line[x];
							if (a<bestval) bestval=a;
							if (b<bestval) bestval=b;
							if (c<bestval) bestval=c;
							if (d<bestval) bestval=d;

							if (bestval&MASKOFF && xorcount!=0) {
								for (i=0; i<g_numimages; i++) {
									if (mask[i]==MASKON) {
										g_seamwarning=true;
										bestval=MASKOFF|i;
										if (!g_reverse) break;
									}
								}
							}

							line[x++]=bestval;

							a=b+(1<<8);
							b=c-(1<<8);
							d=bestval+(3<<8);
						}
					}

					if (lastpixel) {
						testval=VALMASKED(line[-g_workwidth+x])+(3<<8);
						bestval=std::min(line[x],testval);

						testval=VALMASKED(line[-g_workwidth+x-1])+(4<<8);
						if (testval<bestval) bestval=testval;

						testval=VALMASKED(line[x-1])+(3<<8);
						if (testval<bestval) bestval=testval;

						if (bestval&MASKOFF && xorcount!=0) {
							for (i=0; i<g_numimages; i++) {
								if (mask[i]==MASKON) {
									g_seamwarning=true;
									bestval=MASKOFF|i;
									break;
									if (!g_reverse) break;
								}
							}
						}

						line[x++]=bestval;

						lastpixel=false;
					}
				}
			}

			for (i=0; i<g_numimages; i++) maskcount[i]-=mincount;
		}
		y++;
	}
	free(mask);
	free(masklimit);
	free(maskcount);
}

void leftupxy() {
	int i;
	int x,y;
	int xorcount;
	int xorimage;
	int mincount = 0;
	int stop;
	uint32 temp;
	int* maskcount = (int*)malloc(g_numimages*sizeof(int));
	int* masklimit = (int*)malloc(g_numimages*sizeof(int));
	int* mask = (int*)malloc(0x100*sizeof(int));
	bool lastpixel = false;
	uint32* line;
	uint32 bestval,testval;
	uint32 a,b,c,d;

	mask[255] = MASKOFF;

	y = g_workheight - 1;
	while (y >= 0) 
	{
		line = &g_edt[y*g_workwidth];
		// set maskcount, masklimit, binary_mask.pointer
		for (i = 0; i < g_numimages; i++) 
		{
			mask[i] = MASKOFF;
			if (y >= g_images[i].ypos && y < g_images[i].ypos + g_images[i].height) 
			{
				maskcount[i] = g_workwidth - (g_images[i].xpos + g_images[i].width);
				masklimit[i] = g_images[i].xpos;
				g_images[i].binary_mask.pointer = &g_images[i].binary_mask.data[g_images[i].binary_mask.rows[y - g_images[i].ypos + 1]]; // point to END of line
			} 
			else 
			{
				maskcount[i]=g_workwidth;
				masklimit[i]=g_workwidth;
			}
		}

		x = g_workwidth - 1;
		while (x >= 0) 
		{
			mincount = (x + 1); // {0,1,2,3,4} , count = 4 + 1
			xorcount = 0;
			for (i = 0; i < g_numimages; i++) 
			{
				if (maskcount[i] == 0) 
				{
					if (x >= masklimit[i]) 
					{
						PREViMASK(i); //if end of line [ ][ ][ ][ ][ ]. --> [ ][ ][ ][ ].[*], get mask, get maskcount
					}
					else 
					{
						mask[i] = MASKOFF;
						maskcount[i] = mincount;
					}
				}

				if (maskcount[i] < mincount) 
					mincount = maskcount[i];
				if (mask[i] != MASKOFF) 
				{
					xorcount++;
					xorimage = i;
				}
			}

			stop = x - mincount;

			if (xorcount == 1) 
			{
				g_images[xorimage].seampresent = true;
				while (x > stop) 
					line[x--] = xorimage;
			} 
			else 
			{
				// if we're on the bottom line:
				if (y == g_workheight - 1) 
				{
					if (x == g_workwidth - 1) 
					{
						while (x > stop) line[x--] = EDT_MAX;
					} 
					else
					{
						while (x > stop)
						{
							testval = VALMASKED(line[x + 1]) + (3 << 8);
							line[x--] = std::min(testval, EDT_MAX);
						}
					}
				} 
				else 
				{
					// if we're not on the bottom line
					if (x == g_workwidth - 1) 
					{
						testval = VALMASKED(line[+g_workwidth+x])+(3<<8);
						bestval = std::min(EDT_MAX,testval);

						testval = VALMASKED(line[+g_workwidth+x-1])+(4<<8);
						if (testval < bestval) 
							bestval = testval;

						line[x--] = bestval;
					}

					if (stop == -1) 
					{
						stop = 0;
						lastpixel = true;
					}

					/*  xd
					 abc */
					if (x > stop) 
					{
						b = VALMASKED(line[+g_workwidth+x])+(3<<8);
						c = VALMASKED(line[+g_workwidth+x+1])+(4<<8);
						d = VALMASKED(line[x+1])+(3<<8);
					}

					while (x>stop) 
					{ // main bit
						temp = line[+g_workwidth+x-1];
						a = VALMASKED(temp)+(4<<8);

						bestval = EDT_MAX;
						if (a<bestval) bestval = a;
						if (b<bestval) bestval = b;
						if (c<bestval) bestval = c;
						if (d<bestval) bestval = d;

						line[x--] = bestval;

						c = b + (1<<8);
						b = a - (1<<8);
						d = bestval + (3<<8);
					}

					if (lastpixel) 
					{
						testval = VALMASKED(line[+g_workwidth+x])+(3<<8);
						bestval = std::min(EDT_MAX,testval);

						testval = VALMASKED(line[+g_workwidth+x+1])+(4<<8);
						bestval = std::min(bestval,testval);

						testval = VALMASKED(line[x+1]) + (3<<8);
						bestval = std::min(bestval, testval);

						line[x--] = bestval;

						lastpixel=false;
					}
				}
			}

			for (i=0; i<g_numimages; i++) 
			{
				maskcount[i]-=mincount;
			}
		}
		y--;
	}

	free(mask);
	free(masklimit);
	free(maskcount);
}

void find_seamdistances_cycle_y_horiz(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &outmask, const std::vector<cv::Mat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight)
{
	std::vector<const uint8_t*> pmasks(masks.size(), NULL);
	const uint8_t *poutmask = NULL;
	int* pdist = NULL;
	uint8_t* pmat = NULL;

	for (int y = ybeg; y < yend; ++y)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<uint8_t>(y);
		poutmask = outmask.ptr<uint8_t>(y);
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
			if (pdist[x - shift] + l_straight < pdist[x] && (pmasks[pmat[x - shift]][x] || poutmask[x]))
			{
				pdist[x] = pdist[x - shift] + l_straight;
				pmat[x] = pmat[x - shift];
			}
			x += shift;
		}
	}
}

void find_seamdistances_cycle_x(
	const std::vector<const uint8_t*> &pmasks, const uint8_t *poutmask, int *pdist, int *pdist_prev, uint8_t *pnums, uint8_t *pnums_prev,
	int tmp_xbeg, int tmp_xend,
	int l_straight, int l_diag)
{
	for (int x = tmp_xbeg; x < tmp_xend; ++x)
	{
		if (pdist[x] == 0)
			continue;

		if (pdist_prev[x] + l_straight < pdist[x] && (pmasks[pnums_prev[x]][x] || poutmask[x]))
		{
			pdist[x] = pdist_prev[x] + l_straight;
			pnums[x] = pnums_prev[x];
		}

		if (x != tmp_xbeg)
		{
			if (pdist_prev[x - 1] + l_diag < pdist[x] && (pmasks[pnums_prev[x - 1]][x] || poutmask[x]))
			{
				pdist[x] = pdist_prev[x - 1] + l_diag;
				pnums[x] = pnums_prev[x - 1];
			}
		}

		if (x != (tmp_xend - 1))
		{
			if (pdist_prev[x + 1] + l_diag < pdist[x] && (pmasks[pnums_prev[x + 1]][x] || poutmask[x]))
			{
				pdist[x] = pdist_prev[x + 1] + l_diag;
				pnums[x] = pnums_prev[x + 1];
			}
		}
	}
}

void find_seamdistances_cycle_y_vert(
	cv::Mat &dist, cv::Mat &mat, const cv::Mat &outmask, const std::vector<cv::Mat> &masks,
	int shift, int ybeg, int yend, int xbeg, int xend,
	int l_straight, int l_diag)
{
	std::vector<const uint8_t*> pmasks(masks.size(), NULL);
	const uint8_t *poutmask = NULL;

	int *pdist = NULL;
	uint8_t *pmat = NULL;
	auto pdist_prev = dist.ptr<int>(ybeg - shift);
	auto pmat_prev = mat.ptr<uint8_t>(ybeg - shift);

	int y = ybeg;
	while (y != yend)
	{
		pdist = dist.ptr<int>(y);
		pmat = mat.ptr<uint8_t>(y);
		poutmask = outmask.ptr<uint8_t>(y);

		for (int i = 0; i < g_numimages; ++i)
			pmasks[i] = masks[i].ptr<uint8_t>(y);

		find_seamdistances_cycle_x(pmasks, poutmask, pdist, pdist_prev, pmat, pmat_prev, xbeg, xend, l_straight, l_diag);

		pdist_prev = pdist;
		pmat_prev = pmat;

		y += shift;
	}
}

void init_seamdist(cv::Mat &dist, cv::Mat &nums, cv::Mat &outmask, const std::vector<cv::Mat> &masks)
{
	std::vector<const uint8_t*> pmasks(masks.size());

	for (int y = 0; y < g_workheight; ++y)
	{
		int* pdist = dist.ptr<int>(y);
		uint8_t* pnums = nums.ptr<uint8_t>(y);
		uint8_t* poutmask = outmask.ptr<uint8_t>(y);
		for (int i = 0; i < masks.size(); ++i)
			pmasks[i] = masks[i].ptr<uint8_t>(y);

		for (int x = 0; x < g_workwidth; ++x)
		{
			int count = 0;
			int num = 0;
			for (int i = 0; i < masks.size(); ++i)
				if (pmasks[i][x]) //1-visible
				{
					++count;
					num = i;
				}

			if (count == 0)
				poutmask[x] = 1;
			else
				poutmask[x] = 0;

			if (count == 1)
			{
				pdist[x] = 0;
				pnums[x] = num;
			}
			else
			{
				pdist[x] = 4*(g_workwidth + g_workheight);
				pnums[x] = 0xff;
			}
		}
	}
}

void set_g_edt_opencv(cv::Mat &dist, cv::Mat &nums, cv::Mat &outmask, const std::vector<cv::Mat> &masks)
{
	printf("set_g_edt_opencv\n");
	dist = cv::Mat(g_workheight, g_workwidth, CV_32S);
	nums = cv::Mat(g_workheight, g_workwidth, CV_8U);
	outmask = cv::Mat(g_workheight, g_workwidth, CV_8U);
	
	init_seamdist(dist, nums, outmask, masks);

	int ybeg, yend;
	int xbeg, xend;

//vertical
	xbeg = 0;
	xend = g_workwidth;

	// bottom to top
	ybeg = g_workheight - 1 - 1;
	yend = -1;
	find_seamdistances_cycle_y_vert(dist, nums, outmask, masks, -1, ybeg, yend, xbeg, xend, L_STRAIGHT_SEAM, L_DIAG_SEAM);

	// top to bottom
	//TODO 1:rows --> ypos:ypos+height...
	ybeg = 1;
	yend = g_workheight;
	find_seamdistances_cycle_y_vert(dist, nums, outmask, masks, 1, ybeg, yend, xbeg, xend, L_STRAIGHT_SEAM, L_DIAG_SEAM);

//horizontal
	ybeg = 0;
	yend = g_workheight;
	
	//right to left
	xbeg = (g_workwidth - 1) - 1; // overlap_of_pano_split * ((g_workwidth - 1) - 1);
	xend = -1;
	find_seamdistances_cycle_y_horiz(dist, nums, outmask, masks, -1, ybeg, yend, xbeg, xend, L_STRAIGHT_SEAM);

	//left to right
	xbeg =  1;
	xend = g_workwidth; // overlap_of_pano_split * g_workwidth;
	find_seamdistances_cycle_y_horiz(dist, nums, outmask, masks, 1, ybeg, yend, xbeg, xend, L_STRAIGHT_SEAM);
}

void simple_seam() {

	int i;
	int x,y;
	int p=0;
	int dy;
	int max;
	int best;
	size_t size=(g_numimages*g_workheight)<<2;

	g_seams=(uint32*)malloc(size*sizeof(uint32));

	for (i=0; i<g_numimages; i++) {
		g_images[i].cx=(int)(g_images[i].xpos+g_images[i].width*0.5);
		g_images[i].cy=(int)(g_images[i].ypos+g_images[i].height*0.5);
	}

	for (y=0; y<g_workheight; y++) {
		for (i=0; i<g_numimages; i++) {
			if (g_images[i].ypos<y || g_images[i].ypos+g_images[i].height>=y) { g_images[i].d=-1; continue; }
			g_images[i].dx=-g_images[i].cx;
			dy=g_images[i].cy-y;
			g_images[i].d=g_images[i].cx*g_images[i].cx+dy*dy;
		}

		for (x=0; x<g_workwidth; x++) {
			max=0x7fffffff;
			best=255; // default to a non image
			for (i=0; i<g_numimages; i++) {
				if (g_images[i].d==-1) continue;
				if (g_images[i].d<max) {
					best=i;
					max=g_images[i].d;
				}
				g_images[i].d+=(g_images[i].dx*2)+1;
				g_images[i].dx++;
			}
		}
	}
}

void make_seams() {
	int x,y;
	int p=0;
	size_t size;
	int count=1;
	int a,b;
	uint32* line;

	size=(g_numimages*g_workheight)<<2; // was g_workwidth<<3
	g_seams=(uint32*)malloc(size*sizeof(uint32));

	for (y=0; y<g_workheight; y++) {
		line=&g_edt[y*g_workwidth];
		a=line[0]&0xff; // number of image
		x=1;
		while (x<g_workwidth) {
			b=line[x++]&0xff;
			if (b!=a) {
				g_seams[p++] = count<<8|a;
				count=1;
				a=b;
			} else {
				count++;
			}
		}
		g_seams[p++]=count<<8|a;
		count=1;

//		if ((p+(g_workwidth<<3))>size) {
//			size+=g_workwidth<<4;
		if ((size_t)(p+((g_numimages*g_workheight)<<1))>size) {
			size+=(g_numimages*g_workheight)<<2;
			g_seams=(uint32*)realloc(g_seams,size*sizeof(uint32));
		}
	}

	g_seams=(uint32*)realloc(g_seams,p*sizeof(uint32));
}

void write_g_edt()
{
	cv::Mat mat_mask(g_workheight, g_workwidth, CV_8U);
	cv::Mat mat_dist(g_workheight, g_workwidth, CV_8U);
	cv::Mat mat_image(g_workheight, g_workwidth, CV_8U);
	int p = 0;
	for (int y = 0; y < g_workheight; ++y)
	{
		for (int x = 0; x < g_workwidth; ++x)
		{
			uint32 tmp = g_edt[p];
			mat_mask.at<uint8_t>(y, x) = (tmp >> 31) * 255;
			mat_dist.at<uint8_t>(y, x) = ((tmp >> 8) & 0xffff) / 10;
			mat_image.at<uint8_t>(y, x) = (tmp & 0xf) * 30;
			++p;
		}
	}
	cv::imwrite("seam_mask.png", mat_mask);
	cv::imwrite("seam_dist.png", mat_dist);
	cv::imwrite("seam_image.png", mat_image);
}

void seam() {
	int i;

	output(1,"seaming...\n");
	for (i=0; i<g_numimages; i++) g_images[i].seampresent=false;

	if (!g_seamload_filename) {
		if (g_xor_filename) seam_png(0,g_xor_filename);
		#ifdef NO_OPENCV
			if (!g_simpleseam) {
				g_edt=(uint32*)_aligned_malloc(g_workwidth*g_workheight*sizeof(uint32),0); // if malloc fails fall back on dtcomp
				if (!g_edt) die("not enough memory to create seams");
				leftupxy();
				rightdownxy();
				make_seams();
				_aligned_free(g_edt);
				/*
				cv::imwrite("seam_image_opencv.png", g_cvseams * 30);
				cv::Mat tmp;
					dist /= 10;
				dist.convertTo(tmp, CV_8U);
				cv::imwrite("seam_dist_opencv.png", tmp);
				dist *= 10;
				write_g_edt();
			
				cv::Mat sum = g_cvmasks[0].clone();
				sum /= 10;
				for (int i = 1; i < g_numimages; ++i)
					sum += g_cvmasks[i] / 10;
				cv::Mat newsum(sum, cv::Rect(0,0,g_workwidth, g_workheight));
				cv::imwrite("sum_masks.png", newsum);
				*/
			} else {
				simple_seam();
			}
			if (g_seamsave_filename) seam_png(1,g_seamsave_filename);
		#else
		if (g_cvmaskpyramids.empty() || g_cvoutmask.empty())
		{
			cv::Mat dist;
			set_g_edt_opencv(dist, g_cvseams, g_cvoutmask, g_cvmasks);
		}
		#endif

		for (i=0; i<g_numimages; i++) {
			if (!g_images[i].seampresent) {
				printf("WARNING: some images completely overlapped\n");
				break;
			}
		}
		if (g_seamwarning) printf("WARNING: some image areas have been arbitrarily assigned\n");
	} else {
		load_seams();

		for (i=0; i<g_numimages; i++) {
			if (!g_images[i].seampresent) {
				printf("WARNING: some images not present in seam bitmap\n");
				break;
			}
		}
	}
}
