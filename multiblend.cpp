/*
	multiblend (c) 2013 David Horman

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2, or (at your option)
	any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.

	The author can be contacted at: david.horman@jerseymail.co.uk

	Discussion at http://tawbaware.com/forum2/viewtopic.php?f=3&t=6396
						 or http://groups.google.com/group/hugin-ptx/
*/
#include "multiblend.h"

#include "globals.h"
#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <algorithm>
#include <stdio.h>

#ifdef WIN32
#pragma comment(lib,"libtiff.lib")
#pragma comment(lib,"turbojpeg-static.lib")
#pragma comment(lib,"libpng.lib")
#pragma comment(lib,"zlib.lib")
#endif

void help() {
	printf("Usage: multiblend [options] [-o OUTPUT] INPUT...\n");
	printf("Options:\n");
	printf("   -l X                  X > 0: limit number of blending levels to x\n");
	printf("                         X < 0: reduce number of blending levels by -x\n");
	printf("   -d DEPTH              override automatic output image depth (8 or 16)\n");
	printf("  --nocrop               do not crop output\n");
	printf("  --bgr                  swap RGB order\n");
	printf("  --wideblend            calculate number of levels based on output image size,\n");
	printf("                         rather than input image size\n");
	printf("  --compression=X        output file compression. For TIFF output, X may be:\n");
	printf("                         NONE (default), PACKBITS, or LZW\n");
	printf("                         For JPEG output, X is JPEG quality (0-100, default 75)\n");
	printf("  --cache                cache input images to disk to minimise memory usage\n");
	printf("  --save-seams <file>    Save seams to PNG file for external editing\n");
	printf("  --no-output            Don't perform blend (for use with --save-seams)\n");
	printf("  --load-seams <file>    Load seams from PNG file\n");
	printf("  --bigtiff              BigTIFF output (not well tested)\n");
	printf("  --reverse              reverse image priority (last=highest)\n");
	printf("  --quiet                suppress output (except warnings)\n");
	printf("\n");
	printf("Pass a single image as input to blend around the left/right boundary.\n");
	exit(0);
}

void parse(std::vector<std::string> &output, const std::string &input)
{
	std::stringstream os(input);
	std::string temp;
	while (os >> temp)
		output.push_back(temp);
}

int multiblend(const std::string &inputstring, const std::vector<cv::Mat> &mats, const std::vector<cv::Mat> &masks) {
	Proftimer proftimer(&mprofiler, "root");

	#if TIFF_LIBRARY
	TIFFSetWarningHandler(NULL);
	#endif

	std::vector<std::string> args;
	args.push_back("multiblend");
	parse(args, inputstring);
	int argc = args.size();
	const char **argv = new const char*[argc];
	for (int i = 0; i < argc; ++i)
		argv[i] = args[i].c_str();

	for (int i = 0; i < argc; ++i)
	{
		printf("argv[%d] = %s\n", i, argv[i]);
	}

	int temp;

	printf("\n");
	printf("multiblend v0.6.2 (c) 2014 David Horman        http://horman.net/multiblend/\n");
	printf("----------------------------------------------------------------------------\n");

	if (argc==1 || !strcmp(argv[1],"-h") || !strcmp(argv[1],"--help") || !strcmp(argv[1],"/?")) help();

	printf("\n");

	if (argc<3) die("not enough arguments (try -h for help)");

	for (int i=1; i<argc-1; i++) {
		if (!strcmp(argv[i],"-d")) {
			g_workbpp_cmd=atoi(argv[++i]);
			if (g_workbpp_cmd!=8 && g_workbpp_cmd!=16) {
				die("invalid output depth specified");
			}
		}
		else if (!strcmp(argv[i],"-l")) {
			temp=atoi(argv[++i]);
			if (temp>=0) g_max_levels=std::max(1,temp); else g_sub_levels=-temp;
		}
		else if (!strcmp(argv[i],"--nomask")) g_nomask=true;
		else if (!strcmp(argv[i],"--nocrop")) g_crop=false;
		else if (!strcmp(argv[i],"--bigtiff")) g_bigtiff=true;
		else if (!strcmp(argv[i],"--bgr")) g_bgr=true;
		else if (!strcmp(argv[i],"--wideblend")) g_wideblend=true;
		else if (!strcmp(argv[i],"--noswap")) g_swap=false;
		else if (!strcmp(argv[i],"--reverse")) g_reverse=true;
		else if (!strcmp(argv[i],"--timing")) g_timing=true;
		else if (!strcmp(argv[i],"--dewhorl")) g_dewhorl=true;

		else if (!strcmp(argv[i],"-w")) output(0,"ignoring enblend option -w\n");
		else if (!strncmp(argv[i],"-f",2)) output(0,"ignoring enblend option -f\n");
		else if (!strcmp(argv[i],"-a")) output(0,"ignoring enblend option -a\n");
		else if (!strcmp(argv[i],"--no-ciecam")) output(0,"ignoring enblend option --no-ciecam\n");
		else if (!strncmp(argv[i],"--primary-seam-generator",24)) output(0,"ignoring enblend option --primary-seam-generator\n");

		else if (!strncmp(argv[i],"--compression",13)) {
			char* comp=(char*)argv[i]+14;
			if (strcmp(comp,"0")==0) g_jpegquality=0;
			else if (atoi(comp)>0) g_jpegquality=atoi(comp);
			#if TIFF_LIBRARY
			else if (_stricmp(comp,"lzw")==0) g_compression=COMPRESSION_LZW;
			else if (_stricmp(comp,"packbits")==0) g_compression=COMPRESSION_PACKBITS;
			else if (_stricmp(comp,"none")==0) g_compression=COMPRESSION_NONE;
			#endif
 			//else if (_stricmp(comp,"deflate")==0) g_compression=COMPRESSION_DEFLATE;
			else die("unknown compression codec");
		}
		else if (!strcmp(argv[i],"-v") || !strcmp(argv[i],"--verbose")) g_verbosity++;
		else if (!strcmp(argv[i],"-q") || !strcmp(argv[i],"--quiet")) g_verbosity--;
		else if (!strcmp(argv[i],"--debug")) g_debug=true;
		else if (!strcmp(argv[i],"--saveseams") || !strcmp(argv[i],"--save-seams")) g_seamsave_filename=(char*)argv[++i];
		else if (!strcmp(argv[i],"--loadseams") || !strcmp(argv[i],"--load-seams")) g_seamload_filename=(char*)argv[++i];
		else if (!strcmp(argv[i],"--simpleseam") || !strcmp(argv[i],"--simple-seam")) g_simpleseam=true;
		else if (!strcmp(argv[i],"--savemasks") || !strcmp(argv[i],"--save-masks")) g_savemasks=true;
		else if (!strcmp(argv[i],"--saveoutpyramids")) g_save_out_pyramids=true;
		else if (!strcmp(argv[i],"--savexor") || !strcmp(argv[i],"--save-xor")) g_xor_filename=(char*)argv[++i];
		else if (!strcmp(argv[i],"--no-output")) g_nooutput=true;
		else if (!strcmp(argv[i],"--cache")) g_caching=true;
		else if (!strcmp(argv[i],"-o") || !strcmp(argv[i],"--output")) {
			g_output_filename=(char*)argv[++i];
			char* ext=strrchr(g_output_filename,'.')+1;
			if (!(_stricmp(ext,"jpg") && _stricmp(ext,"jpeg"))) {
				if (g_compression!=-1) {
					output(0,"warning: JPEG output; ignoring TIFF compression setting\n");
					g_compression=-1;
				}
				if (g_jpegquality==-1) g_jpegquality=75;
			} else if (!(_stricmp(ext,"tif") && _stricmp(ext,"tiff"))) {
				if (g_jpegquality!=-1) {
					output(0,"warning: TIFF output; ignoring JPEG quality setting\n");
					g_jpegquality=-1;
				}
				#if TIFF_LIBRARY
				else if (g_compression==-1) g_compression=COMPRESSION_NONE;
				#endif
			} else {
				die("unknown file extension!");
			}

			i++;
			break;
		} else {
			die ("unknown argument \"%s\"",argv[i]);
		}
	}

	if (!g_output_filename && !g_seamsave_filename) die("no output file specified");

	go(mats, masks);
	clear_temp();

	if (g_debug) {
		printf("\nPress Enter to end\n");
		getchar();
	}

	proftimer.stop();
	mprofiler.report();
	delete[] argv;

	return 0;
}
/*
int main()
{
	std::string inputstring = "-d 8 --wideblend --nocrop -o ./000000001.tif";
	std::vector<cv::Mat> mats;
	std::vector<cv::Mat> masks;
	for (int i  = 1; i <= 6; ++i)
	{
		char buf[128];
		sprintf(buf,"000000001-%d.tif", i);
		mats.push_back(cv::imread(buf, CV_LOAD_IMAGE_COLOR));
		masks.push_back(cv::imread(std::string("mask_") + std::string(buf),CV_LOAD_IMAGE_GRAYSCALE));
	}
	return multiblend(inputstring, mats, masks);
}
*/
