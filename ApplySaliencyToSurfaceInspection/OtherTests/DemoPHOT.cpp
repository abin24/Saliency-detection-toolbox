//paper 26
// This script are used to re - produce the PHOT approach for anomaly
// detection mentioned in the article
// By Yibin huang
//"The Phase Only Transform for unsupervised surface defect detection", Aiger & Talbot 2010.
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
 
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	Mat src = imread("sample.bmp", 1);
	imshow("src",src);
	Mat gray,grayTemp, grayDown;
	vector<Mat> mv;
	cvtColor(src, gray, CV_BGR2GRAY);

	Size imageSize(gray.cols , gray.rows);
	Mat realImage(imageSize, CV_64F);
	Mat imaginaryImage(imageSize, CV_64F); imaginaryImage.setTo(0);
	Mat combinedImage(imageSize, CV_64FC2);
	Mat imageDFT;
	Mat logAmplitude;
	Mat angle(imageSize, CV_64F);
	Mat magnitude(imageSize, CV_64F);
	Mat logAmplitude_blur;

	gray.convertTo(realImage,CV_64F);

	mv.push_back(realImage);
	mv.push_back(imaginaryImage);
	merge(mv, combinedImage);
	dft(combinedImage, imageDFT);
	split(imageDFT, mv);

	//-- Get magnitude and phase of frequency spectrum --//
	cartToPolar(mv.at(0), mv.at(1), magnitude, angle, false);
	log(magnitude, logAmplitude);
	//-- Blur log amplitude with averaging filter --//
	blur(logAmplitude, logAmplitude_blur, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
	exp(logAmplitude - logAmplitude_blur, magnitude);
	//-- Back to cartesian frequency domain --//
	polarToCart(magnitude, angle, mv.at(0), mv.at(1), false);
	merge(mv, imageDFT);

	dft(imageDFT, combinedImage, CV_DXT_INVERSE);//invert dft
	split(combinedImage, mv);

	cartToPolar(mv.at(0), mv.at(1), magnitude, angle, false);
	GaussianBlur(magnitude, magnitude, Size(5, 5), 8, 0, BORDER_DEFAULT);
	magnitude = magnitude.mul(magnitude);

	double minVal, maxVal;
	minMaxLoc(magnitude, &minVal, &maxVal);
	magnitude = magnitude / maxVal;//normalize	 
 
	imshow("SR by XiaodiHou", magnitude>0.8);




	vector<Mat> Rp;
	Mat Real(imageSize, CV_64F);
	Mat phase(imageSize, CV_64F); imaginaryImage.setTo(0);
	Mat DFT,RI; 
	gray.convertTo(Real, CV_64F);	 
	Rp.push_back(Real);
	Rp.push_back(phase);	
	merge(Rp, RI);
	dft(RI, DFT);
	split(DFT, Rp);
	Mat MAG, PHASE;
	cartToPolar(Rp.at(0), Rp.at(1), MAG, PHASE, false);
	cout << "First" << endl;
	cout << Rp.at(0).at<double>(0, 0) << endl;
	cout << Rp.at(1).at<double>(0, 0) << endl;
	cout << MAG.at<double>(0, 0);



	Rp.at(0) = Rp.at(0)/MAG;
	Rp.at(1) = Rp.at(1)/ MAG;
	merge(Rp, RI);
//	DFT=DFT / RI;
	//split(DFT, Rp);
	cout << "second" << endl;
	cout << Rp.at(0).at<double>(0, 0) << endl;
	cout << Rp.at(1).at<double>(0, 0) << endl;
	//cout << RI;
	dft(RI, RI, CV_DXT_INVERSE_SCALE);//invert dft
	
	split(RI, Rp);
	/*cartToPolar(Rp.at(0), Rp.at(1), MAG, PHASE, false);
	cout << Rp.at(0).at<double>(0, 0) << endl;
	cout << Rp.at(1).at<double>(0, 0) << endl;
	cout << MAG.at<double>(0, 0);*/
	MAG = Rp.at(0);
	GaussianBlur(MAG, MAG, Size(7, 7), 0);
	Scalar MeanS = mean(MAG);
	double m = MeanS[1];
	
	 
	for (int r = 0; r < MAG.rows; r++)
	{
		double *s = MAG.ptr<double>(r);
		
		for (int c = 0; c < MAG.cols; c++)
			s[c] = (s[c] - m)*(s[c] - m);
	}
	double minval, maxval;
	minMaxLoc(MAG, &minval, &maxval);
	MAG = MAG / maxval;//normaliz
	 
	imshow("MAG", MAG>0.9);
	waitKey(0);
	return(0);
}

/** @function cornerHarris_demo */
 