#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo(int, void*);

/** @function main */
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	src = imread("Scratch_6.bmp", 1);
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window and a trackbar
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
	imshow(source_window, src);
 
	cornerHarris_demo(thresh, 0);
	Mat img;
	cvtColor(src,img,CV_RGB2GRAY);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;
	Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("【效果图】 X方向Sobel", abs_grad_x);

	//【4】求Y方向梯度
	Sobel(img, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("【效果图】Y方向Sobel", abs_grad_y);

	//【5】合并梯度(近似)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	imshow("【效果图】整体方向Sobel", dst);
	waitKey(0);
	return(0);
}

/** @function cornerHarris_demo */
void cornerHarris_demo(int, void*)
{

	Mat dst, dstenhance,dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);
	//dstenhance = Mat::zeros(src.size(), CV_32FC1);
	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 4;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing	 
	 
	Scalar meanS =    mean(dst);
	dstenhance = (dst - meanS[0]).mul(dst - meanS[0]);
	normalize(dstenhance, dst_norm, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	//cout << dst_norm << endl;
	//dst_norm_scaled = dst_norm > thresh;
	 
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dstenhance);
	imshow("dst", dst_norm);
}