#ifndef MULTIBLEND_H
#define MULTIBLEND_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>

int multiblend(const std::string &inputstring, std::vector<cv::Mat> &mats, std::vector<cv::Mat> &masks, std::vector<std::vector<cv::Mat> > &cvmaskpyramids, cv::Mat &cvoutmask);

#endif
