#ifndef MULTIBLEND_H
#define MULTIBLEND_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

int multiblend(const std::string &inputstring, std::vector<cv::Mat> &mats, std::vector<cv::Mat> &masks);

#endif
