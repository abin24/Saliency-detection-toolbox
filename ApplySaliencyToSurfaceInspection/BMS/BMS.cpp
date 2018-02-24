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

#include "BMS.h"

#include <vector>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

#define COV_MAT_REG 50.0f


void BMS::BMSinit(const Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening)
	
{
	mDilationWidth_1=dw1;
	mNormalize=nm;
	mHandleBorder=hb;
	mAttMapCount=0;
	mColorSpace=colorSpace;
	mWhitening=whitening;
	mSrc = src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);

	if (CL_RGB & colorSpace)
		whitenFeatMap(mSrc, COV_MAT_REG);
	if (CL_Lab & colorSpace)
	{
		Mat lab;
		cvtColor(mSrc, lab, CV_RGB2Lab);
		whitenFeatMap(lab, COV_MAT_REG);
	}
	if (CL_Luv & colorSpace)
	{
		Mat luv;
		cvtColor(mSrc, luv, CV_RGB2Luv);
		whitenFeatMap(luv, COV_MAT_REG);
	}
}


// calculateSaliencyMap(Mat *src, Mat * dst, int sample_step = 8,int dw1 = 7, bool nm = 1, bool hb = 0, int colorSpace = 2, bool whitening = 1, float max_dimension = -1);
void BMS::calculateSaliencyMap(cv::Mat *src, cv::Mat * dst, int sample_step, int dw1, bool nm, bool hb, int colorSpace, bool whitening, float max_dimension)
{
	 
	BMS bms;
	bms.BMSinit(*src, dw1, nm, hb, colorSpace, whitening);
	bms.computeSaliency((double)sample_step);
	*dst = bms.getSaliencyMap();
 
}
void BMS::computeSaliency(double step)
{
	for (int i=0;i<mFeatureMaps.size();++i)
	{
		Mat bm;
		double max_,min_;
		minMaxLoc(mFeatureMaps[i],&min_,&max_);
		for (double thresh = min_; thresh < max_; thresh += step)
		{
			bm=mFeatureMaps[i]>thresh;
			Mat am = getAttentionMap(bm, mDilationWidth_1, mNormalize, mHandleBorder);
			mSaliencyMap += am;
			mAttMapCount++;
		}
	}
}


cv::Mat BMS::getAttentionMap(const cv::Mat& bm, int dilation_width_1, bool toNormalize, bool handle_border) 
{
	Mat ret=bm.clone();
	int jump;
	if (handle_border)
	{
		for (int i=0;i<bm.rows;i++)
		{
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(i,0+jump)!=1)
				floodFill(ret,Point(0+jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
			jump = BMS_RNG.uniform(0.0,1.0)>0.99 ?BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(i,bm.cols-1-jump)!=1)
				floodFill(ret,Point(bm.cols-1-jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
		for (int j=0;j<bm.cols;j++)
		{
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(0+jump,j)!=1)
				floodFill(ret,Point(j,0+jump),Scalar(1),0,Scalar(0),Scalar(0),8);
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(bm.rows-1-jump,j)!=1)
				floodFill(ret,Point(j,bm.rows-1-jump),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
	}
	else
	{
		for (int i=0;i<bm.rows;i++)
		{
			if (ret.at<uchar>(i,0)!=1)
				floodFill(ret,Point(0,i),Scalar(1),0,Scalar(0),Scalar(0),8);
			if (ret.at<uchar>(i,bm.cols-1)!=1)
				floodFill(ret,Point(bm.cols-1,i),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
		for (int j=0;j<bm.cols;j++)
		{
			if (ret.at<uchar>(0,j)!=1)
				floodFill(ret,Point(j,0),Scalar(1),0,Scalar(0),Scalar(0),8);
			if (ret.at<uchar>(bm.rows-1,j)!=1)
				floodFill(ret,Point(j,bm.rows-1),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
	}
	
	ret = ret != 1;
	
	Mat map1, map2;
	map1 = ret & bm;
	map2 = ret & (~bm);

	if (dilation_width_1 > 0)
	{
		dilate(map1, map1, Mat(), Point(-1, -1), dilation_width_1);
		dilate(map2, map2, Mat(), Point(-1, -1), dilation_width_1);
	}
		
	map1.convertTo(map1,CV_32FC1);
	map2.convertTo(map2,CV_32FC1);

	if (toNormalize)
	{
		normalize(map1, map1, 1.0, 0.0, NORM_L2);
		normalize(map2, map2, 1.0, 0.0, NORM_L2);
	}
	else
		normalize(ret,ret,0.0,1.0,NORM_MINMAX);
	return map1+map2;
}

Mat BMS::getSaliencyMap()
{
	Mat ret; 
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret,CV_8UC1);
	return ret;
}

void BMS::whitenFeatMap(const cv::Mat& img, float reg)
{
	assert(img.channels() == 3 && img.type() == CV_8UC3);
	
	vector<Mat> featureMaps;
	
	if (!mWhitening)
	{
		split(img, featureMaps);
		for (int i = 0; i < featureMaps.size(); i++)
		{
			normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
			medianBlur(featureMaps[i], featureMaps[i], 3);
			mFeatureMaps.push_back(featureMaps[i]);
		}
		return;
	}

	Mat srcF,meanF,covF;
	img.convertTo(srcF, CV_32FC3);
	Mat samples = srcF.reshape(1, img.rows*img.cols);
	calcCovarMatrix(samples, covF, meanF, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);

	covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
	SVD svd(covF);
	Mat sqrtW;
	sqrt(svd.w,sqrtW);
	Mat sqrtInvCovF = svd.u * Mat::diag(1.0/sqrtW);

	Mat whitenedSrc = srcF.reshape(1, img.rows*img.cols)*sqrtInvCovF;
	whitenedSrc = whitenedSrc.reshape(3, img.rows);
	
	split(whitenedSrc, featureMaps);

	for (int i = 0; i < featureMaps.size(); i++)
	{
		normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
		featureMaps[i].convertTo(featureMaps[i], CV_8U);
		medianBlur(featureMaps[i], featureMaps[i], 3);
		mFeatureMaps.push_back(featureMaps[i]);
	}
}