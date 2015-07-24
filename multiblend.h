#ifndef MULTIBLEND_H
#define MULTIBLEND_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>

#ifdef NO_CUDA
int multiblend(const std::string &inputstring, std::vector<cv::Mat> &mats, std::vector<cv::Mat> &masks, std::vector<std::vector<cv::Mat> > &cvmaskpyramids, cv::Mat &cvoutmask);
#else
int multiblend(const std::string &inputstring, std::vector<cv::cuda::GpuMat> &mats, std::vector<cv::cuda::GpuMat> &masks, std::vector<std::vector<cv::cuda::GpuMat> > &cvmaskpyramids, cv::cuda::GpuMat &cvoutmask);
#endif

#endif
