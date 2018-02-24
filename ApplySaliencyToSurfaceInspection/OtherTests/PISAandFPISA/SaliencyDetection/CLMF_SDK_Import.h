#ifndef CLMF_SDK_HEADER
#define CLMF_SDK_HEADER

#define DllImport __declspec(dllimport) 

#include "opencv2\opencv.hpp"
// -------------------------------------------
// option:
// 1 - use filter for color image
//     the input "imgGuided" and "imgIn" are types of cv::Vec3b
//     the output "imgOut" is type of cv::Vec3b
// 2 - use filter for cost slice filtering (e.g. in stereo matching)
//     the input "imgGuided" is type of cv::Vec3b, and "imgIn" is types of float
//     the output "imgOut" is type of float
// -------------------------------------------
// filterOrder:
// 0 - to use CLMF-0 zero-order filter
// 1 - to use CLMF-1 first-order filter -- two times slower
// -------------------------------------------
// imgGuided is the image that use to guide the filtering
// -------------------------------------------
// radius, colorTau, epsl parameters are set as decribed by the CVPR2012 paper
// -------------------------------------------
// ext1:
// 0 - the regular O(r) way to determine cross-based support map
// 1 - a O(1) way to build cross-based support map
// -------------------------------------------
DllImport int FilterImageWithCLMF(
	int option, 
	int filterOrder,
	const cv::Mat &imgGuided, 
	const cv::Mat &imgIn, 
	cv::Mat &imgOut, 
	int radius = 9,
	float colorTau = 25.0,
	float epsl = 0.1*0.1,
	int ext1 = 0);

// -------------------------------------------
// save out the the cross-based support map for evaluation
// 
// the saved-out map images are in format as follows:
// crossMapUpDownImg = (0, up_arm_length, down_arm_length), 
// crossMapLeftRightImg = (left_arm_length, right_arm_length, 0),
// 
// user could multiply the result by a factor 255/L to visualize the maps, 
// L is the maximum arm length
//
// -------------------------------------------
// ext1:
// 0 - the regular O(r) way to determine cross-based support map
// 1 - a O(1) way to build cross-based support map
// -------------------------------------------
DllImport int SaveOutCrossSupportMap(
	const cv::Mat_<cv::Vec3b> &imgGuided, 
	cv::Mat_<cv::Vec3b> &crossMapUpDownImg, 
	cv::Mat_<cv::Vec3b> &crossMapLeftRightImg,
	int maxArmLength = 9,
	float colorTau = 25.0, 
	int ext1 = 0);

#endif