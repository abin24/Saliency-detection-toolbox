//paper 26
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
	src = imread("exp4_num_32185.jpg", 1);
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window and a trackbar
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
	imshow(source_window, src);

	cornerHarris_demo(thresh, 0);
	
	waitKey(0);
	return(0);
}

/** @function cornerHarris_demo */
void cornerHarris_demo(int, void*)
{

	Mat dst, dstenhance, dst_norm, dst_norm_scaled;
	int dsize = 4;
	 
	Mat img;
	cvtColor(src, img, CV_RGB2GRAY);
	resize(img, img, Size(img.cols / dsize, img.rows / dsize));
	Mat Ix, Iy;	 
	Sobel(img, Ix, CV_32F, 1, 0, 3, 1, 1, BORDER_DEFAULT); 
	Sobel(img, Iy, CV_32F, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	Mat Ix2, Iy2, Ixy;
	
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);
	 
	//https://www.cnblogs.com/polly333/p/5416172.html
	 
	GaussianBlur(Ix2, Ix2, Size(5, 5), 0);
	GaussianBlur(Iy2, Iy2, Size(5, 5), 0);
	GaussianBlur(Ixy, Ixy, Size(5, 5), 0);
	Mat A, B;
	A = Ix2 - Iy2;
	
	A = A.mul(A) + 4 * Ixy.mul(Ixy);	
	B = Ix2 + Iy2;
      
	normalize(A/10000000, A, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(B/10000000, B, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
 
	imshow("A",A);
	imshow("B", B);
	 
 
	addWeighted(A, 0.5, B, 0.5, 0, dst);
	resize(dst, dst, Size(img.cols * dsize, img.rows * dsize));
	imshow("Contours", dst);



}