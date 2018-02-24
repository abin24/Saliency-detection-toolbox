/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Exploit Surroundedness for Saliency Detection: A Boolean Map Approach",
*   Jianming Zhang, Stan Sclaroff, submitted to PAMI, 2014
*
*	Copyright (C) 2014 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
*******************************************************************************/
#define MAX_IMG_DIM 400
#ifndef BMS_H
#define BMS_H


#ifdef IMDEBUG
#include <imdebug.h>
#endif
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

static const int CL_RGB = 1;
static const int CL_Lab = 2;
static const int CL_Luv = 4;

static cv::RNG BMS_RNG;

class BMS
{
public:
	static void calculateSaliencyMap(cv::Mat *src, cv::Mat * dst, int sample_step = 3, int dw1 = 3, bool nm = 1, bool hb = 0, int colorSpace = 2, bool whitening = 1, float max_dimension = -1);
public:
	void BMSinit (const cv::Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening);
	cv::Mat getSaliencyMap();
	void computeSaliency(double step);
private:
	cv::Mat mSaliencyMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	std::vector<cv::Mat> mFeatureMaps;
	int mDilationWidth_1;
	bool mHandleBorder;
	bool mNormalize;
	bool mWhitening;
	int mColorSpace;
	cv::Mat getAttentionMap(const cv::Mat& bm, int dilation_width_1, bool toNormalize, bool handle_border);
	void whitenFeatMap(const cv::Mat& img, float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};

void postProcessByRec8u(cv::Mat& salmap, int kernelWidth);
void postProcessByRec(cv::Mat& salmap, int kernelWidth);



#endif


