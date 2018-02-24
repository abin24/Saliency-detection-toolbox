/*************************************************

Copyright: Guangyu Zhong all rights reserved

Author: Guangyu Zhong

Date:2014-09-27

Description: codes for Manifold Ranking Saliency Detection
Reference http://ice.dlut.edu.cn/lu/Project/CVPR13[yangchuan]/cvprsaliency.htm

**************************************************/
#include<iostream>
#include<cv.h>
#include<highgui.h>
#include"GMR.h"
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	Mat img = imread("timg.jpg");
	Mat dst;
	GMR SpMat;
	/*Mat superpixels = SpMat.GeneSp(img);
	Mat sal = SpMat.GeneSal(img);
	 
	Mat salMap = SpMat.Sal2Img(superpixels, sal);
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(tmpsuperpixels, CV_8UC1, 1.0);
	imshow("sp", tmpsuperpixels);*/
	SpMat.calculateSaliencyMap(&img,&dst);
	imshow("sp", dst); 
	waitKey();
	return 0;
}
