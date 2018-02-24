/*************************************************

Copyright: Guangyu Zhong all rights reserved

Author: Guangyu Zhong

Date:2014-09-27

Description: codes for Manifold Ranking Saliency Detection
Reference http://ice.dlut.edu.cn/lu/Project/CVPR13[yangchuan]/cvprsaliency.htm

**************************************************/
#ifndef _GMR_H
#define _GMR_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include "SLIC.h"
using namespace cv;
class GMR
{
public:
	GMR();
	~GMR();
	Mat GeneSal(const Mat &img);
	Mat Sal2Img(const Mat &superpixels, const Mat &Sal);
	Mat GeneSp(const Mat &img);
	void calculateSaliencyMap(Mat *img, Mat* dst);
private:
	int spNumMax;
	double compactness;
	float alpha;
	float theta;
	int spNum;
	Mat GeneAdjMat(const Mat &spmat);
	int GeneFeature(const Mat &img, const Mat &superpixels, const int feaType, Mat &feaSpL, Mat &feaSpA, Mat &feaSpB, Mat &spNpix, Mat &spCnt);
	Mat GeneWeight(const vector<float> &feaSpL, const vector<float> &feaSpA, const vector<float> &feaSpB, const Mat &superpixels, const vector<int> &bd, const Mat &adj);
	vector<int> GeneBdQuery(const Mat &superpixels, const int type);
	Mat GeneY(const vector<int> &bd);
	Mat inveMat(const Mat &weight, const Mat &Y);
	//Mat GeneFeature(const Mat &img);
};


#endif
