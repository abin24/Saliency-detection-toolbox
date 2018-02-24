/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Minimum Barrier Salient Object Detection at 80 FPS", Jianming Zhang,
*	Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price, Radomir Mech, ICCV,
*       2015
*
*	Copyright (C) 2015 Jianming Zhang
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
*	If you have problems about this software, please contact:
*       jimmie33@gmail.com
*******************************************************************************/

#pragma once

#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <cmath>
using namespace std;
using namespace cv;
static cv::RNG MBS_RNG;

class MBP
{
public:
	MBP();
	MBP(const cv::Mat& src);
	void calculateSaliencyMap(cv::Mat*img, cv::Mat* dst);
	cv::Mat getSaliencyMap();
	void computeSaliency(bool use_geodesic = false);
	cv::Mat getMBSMap() const { return mMBSMap; }
	void rasterScan(const cv::Mat& featMap, cv::Mat& map, cv::Mat& lb, Mat& ub);

	void invRasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub);
	float getThreshForGeo(const Mat& src);
	 
	void rasterScanGeo(const Mat& featMap, Mat& map, float thresh);
	void invRasterScanGeo(const Mat& featMap, Mat& map, float thresh);

	cv::Mat computeCWS(const cv::Mat src, float reg, float marginRatio);
	cv::Mat fastMBS(const std::vector<cv::Mat> featureMaps);
	cv::Mat fastGeodesic(const std::vector<cv::Mat> featureMaps);

	int findFrameMargin(const cv::Mat& img, bool reverse);
	bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi);
	Mat doWork(
		const Mat& src,
		bool use_lab,
		bool remove_border,
		bool use_geodesic
		);

	void Reconstruct(Mat src, Mat mask, Mat& dst);
	Mat morpySmooth(Mat I, int radius);
	Mat enhanceConstrast(Mat I, double b = 0.1);
private:
	cv::Mat mSaliencyMap;
	cv::Mat mMBSMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	std::vector<cv::Mat> mFeatureMaps;
	void whitenFeatMap(float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};


 
