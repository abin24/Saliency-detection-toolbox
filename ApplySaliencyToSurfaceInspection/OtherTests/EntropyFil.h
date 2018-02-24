#pragma once
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;
Mat EntropyFiltFn(const Mat& image);

class EntropyFilt
{
public:
	EntropyFilt(const cv::Size& winSize, const cv::Size& blockSize);
	~EntropyFilt();
	Mat Execute(const Mat& image);
	cv::Size blockSize;
private:
	int* neighboodCorrdsArray;
	int* neighboodOffset;
	float* entropyTable;
	cv::Size winSize;
	
};