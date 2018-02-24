//===================================================================================
// Name        : saliencyDetectionRudinac.cpp
// Author      : Joris van de Weem, joris.vdweem@gmail.com
// Version     : 1.1
// Copyright   : Copyright (c) 2011 LGPL
// Description : C++ implementation of "Maja Rudinac, Pieter P. Jonker. 
// 				"Saliency Detection and Object Localization in Indoor Environments". 
//				ICPR'2010. pp.404~407											  
//===================================================================================
// v1.1: Ported to Robot Operating System (ROS)

#include "Rudinac.h"
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

//void Rudinac::imageCB(const sensor_msgs::ImageConstPtr& msg_ptr)
//{
//	cv_bridge::CvImagePtr cv_ptr;
//	sensor_msgs::Image salmap_;
//	geometry_msgs::Point salientpoint_;
//
//	Mat image_, saliencymap_;
//	Point pt_salient;
//	double maxVal;
//
//	try
//	{
//		cv_ptr = cv_bridge::toCvCopy(msg_ptr, enc::BGR8);
//	}
//	catch (cv_bridge::Exception& e)
//	{
//		ROS_ERROR("cv_bridge exception: %s", e.what());
//	}
//	cv_ptr->image.copyTo(image_);
//
//
//	saliencymap_.create(image_.size(),CV_8UC1);
//	Rudinac::calculateSaliencyMap(&image_, &saliencymap_);
//
//	//-- Return most salient point --//
//	cv::minMaxLoc(saliencymap_,NULL,&maxVal,NULL,&pt_salient);
//	salientpoint_.x = pt_salient.x;
//	salientpoint_.y = pt_salient.y;
//
//
//	//	CONVERT FROM CV::MAT TO ROSIMAGE FOR PUBLISHING
//	saliencymap_.convertTo(saliencymap_, CV_8UC1,255);
//	fillImage(salmap_, "mono8",saliencymap_.rows, saliencymap_.cols, saliencymap_.step, const_cast<uint8_t*>(saliencymap_.data));
//
//	saliencymap_pub_.publish(salmap_);
//	point_pub_.publish(salientpoint_);
//
//	return;
//}


void Rudinac::calculateSaliencyMap(const Mat* src, Mat* dst, int corlor )
{
	Size imageSize(128,128);
	Mat srcDown(imageSize,CV_64F);
	Mat magnitudeI(imageSize,CV_64F);
	 
	Mat magnitudeRG(imageSize,CV_64F);
	 
	Mat magnitudeBY(imageSize,CV_64F);
	Mat magnitude(imageSize,CV_64F);
	
	resize(*src, srcDown, imageSize, 0, 0, INTER_LINEAR);
	
	createChannels(&srcDown);
	createSaliencyMap(I,&magnitudeI);
	if (corlor)
	createSaliencyMap(RG,&magnitudeRG);
	if (corlor)
	createSaliencyMap(BY,&magnitudeBY);
	if (corlor)
		magnitude = (magnitudeI + magnitudeRG + magnitudeBY);
	else
		magnitude = magnitudeI;
	GaussianBlur(magnitude, magnitude, Size(5,5), 0, 0, BORDER_DEFAULT);

	//-- Scale to domain [0,1] --//
	double minVal,maxVal;
	minMaxLoc(magnitude, &minVal, &maxVal);
	magnitude = magnitude / maxVal;

	resize(magnitude, *dst, dst->size(), 0, 0, INTER_LINEAR);
}


void Rudinac::createChannels(const Mat* src)
{
	
	b.create(src->size(),CV_32F);
	g.create(src->size(),CV_32F);
	r.create(src->size(),CV_32F);
	I.create(src->size(),CV_32F);
	vector<Mat> planes;	
	split(*src, planes);
	Mat rgmax(src->size(),CV_32F);
	Mat rgbmax(src->size(),CV_32F);
	Mat mask(src->size(),CV_32F);

	for(int j=0; j<r.rows;j++)
       	for(int i=0; i<r.cols; i++)
       	{
			b.at<float>(j,i) = planes[0].at<uchar>(j,i);
			g.at<float>(j,i) = planes[1].at<uchar>(j,i);
			r.at<float>(j,i) = planes[2].at<uchar>(j,i);
       	}
	
	I = r+g+b;
	//threshold(I, I, 255, 255, THRESH_TRUNC); // Saturation as in Matlab?
	I = I/3;
		
	rgmax = max(r,g);
	rgbmax = max(rgmax,b);
	
	//-- Prevent that the lowest value is zero, because you cannot divide by zero.
	for(int j=0; j<r.rows;j++)
       	for(int i=0; i<r.cols; i++)
       	{
			if (rgbmax.at<float>(j,i) == 0) rgbmax.at<float>(j,i) = 1;
       	}


	RG = abs(r-g)/rgbmax;
	BY = abs(b - min(r,g))/rgbmax;

	rgbmax = rgbmax/255;
	//-- If max(r,g,b)<0.1 all components should be zero to stop large fluctuations of the color opponency values at low luminance --//
	threshold(rgbmax,mask,.1,1,THRESH_BINARY);
	RG = RG.mul(mask);
	BY = BY.mul(mask);
	I = I.mul(mask);
}



void Rudinac::createSaliencyMap(const Mat src, Mat* dst)
{
	vector<Mat> mv;	
	
	Mat realImage(src.size(),CV_64F);
	Mat imaginaryImage(src.size(),CV_64F); imaginaryImage.setTo(0);
	Mat combinedImage(src.size(),CV_64FC2);
	Mat image_DFT;	
	Mat logAmplitude;
	Mat angle(src.size(),CV_64F);
	Mat Magnitude(src.size(),CV_64F);
	Mat logAmplitude_blur;
	
	for(int j=0; j<src.rows;j++){
       	for(int i=0; i<src.cols; i++){
       		realImage.at<double>(j,i) = src.at<float>(j,i);
       	}
	}

			
	mv.push_back(realImage);
	mv.push_back(imaginaryImage);	
	merge(mv,combinedImage);	
	
	dft( combinedImage, image_DFT);
	split(image_DFT, mv);	

	//-- Get magnitude and phase of frequency spectrum --//
	cartToPolar(mv.at(0), mv.at(1), Magnitude, angle, false);
	log(Magnitude,logAmplitude);	
		
	//-- Blur log amplitude with averaging filter --//
	blur(logAmplitude, logAmplitude_blur, Size(3,3), Point(-1,-1), BORDER_DEFAULT);
	exp(logAmplitude - logAmplitude_blur,Magnitude);
	
	polarToCart(Magnitude, angle,mv.at(0), mv.at(1),false);
	merge(mv,image_DFT);
	dft(image_DFT,combinedImage,CV_DXT_INVERSE);

	split(combinedImage,mv);
	cartToPolar(mv.at(0), mv.at(1), Magnitude, angle, false);
	Magnitude = Magnitude.mul(Magnitude);

	Mat tempFloat(src.size(),CV_32F);
	for(int j=0; j<Magnitude.rows;j++)
       	for(int i=0; i<Magnitude.cols; i++)
       		tempFloat.at<float>(j,i) = Magnitude.at<double>(j,i);
	
	*dst = tempFloat;
}


 