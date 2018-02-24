//===================================================================================
// Name        : saliencyDetectionHou.cpp
// Author      : Oytun Akman, oytunakman@gmail.com
// Editor	   : Joris van de Weem, joris.vdweem@gmail.com (Conversion to ROS)
// Version     : 1.2
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "Saliency Detection: A Spectral Residual 
//				 Approach" by Xiaodi Hou and Liqing Zhang (CVPR 2007).												  
//===================================================================================
// v1.1: Changed Gaussianblur of logamplitude to averaging blur and gaussian kernel of saliency map to sigma = 8, kernelsize = 5
//      for better consistency with the paper. (Joris)
// v1.2: Ported to Robot Operating System (ROS) (Joris)

#include "SR.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <time.h>
#include<algorithm>



void SR::calculateSaliencyMap(const Mat* src, Mat* dst)
{
	Mat grayTemp, grayDown;
	vector<Mat> mv;	
	//Size imageSize(160,120);
	Size imageSize((*src).cols , (*src).rows );
	Mat realImage(imageSize,CV_64F);
	Mat imaginaryImage(imageSize,CV_64F); imaginaryImage.setTo(0);
	Mat combinedImage(imageSize,CV_64FC2);
	Mat imageDFT;	
	Mat logAmplitude;
	Mat angle(imageSize,CV_64F);
	Mat magnitude(imageSize,CV_64F);
	Mat logAmplitude_blur;
	
	cvtColor(*src, grayTemp, CV_BGR2GRAY);
	resize(grayTemp, grayDown, imageSize, 0, 0, INTER_LINEAR);
	for(int j=0; j<grayDown.rows;j++)
       	for(int i=0; i<grayDown.cols; i++)
       		realImage.at<double>(j,i) = grayDown.at<uchar>(j,i);
			
	mv.push_back(realImage);
	mv.push_back(imaginaryImage);	
	merge(mv,combinedImage);	
	dft( combinedImage, imageDFT);
	split(imageDFT, mv);	

	//-- Get magnitude and phase of frequency spectrum --//
	cartToPolar(mv.at(0), mv.at(1), magnitude, angle, false);
	log(magnitude,logAmplitude);	
	//-- Blur log amplitude with averaging filter --//
	blur(logAmplitude, logAmplitude_blur, Size(3,3), Point(-1,-1), BORDER_DEFAULT);
	
	exp(logAmplitude - logAmplitude_blur,magnitude);
	//-- Back to cartesian frequency domain --//
	polarToCart(magnitude, angle, mv.at(0), mv.at(1), false);
	merge(mv, imageDFT);
	dft( imageDFT, combinedImage, CV_DXT_INVERSE); 
	split(combinedImage, mv);

	cartToPolar(mv.at(0), mv.at(1), magnitude, angle, false);
	GaussianBlur(magnitude, magnitude, Size(5,5), 8, 0, BORDER_DEFAULT);
	magnitude = magnitude.mul(magnitude);

	double minVal,maxVal;
	minMaxLoc(magnitude, &minVal, &maxVal);
	magnitude = magnitude / maxVal;//normalize

	Mat tempFloat(imageSize,CV_32F);
	
	for(int j=0; j<magnitude.rows;j++)
       	for(int i=0; i<magnitude.cols; i++)
       		tempFloat.at<float>(j,i) = magnitude.at<double>(j,i);
	 
 	resize(tempFloat, *dst, dst->size(), 0, 0, INTER_LINEAR);
}
 


	

 