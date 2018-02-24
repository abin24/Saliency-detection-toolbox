//===================================================================================
// Name        : saliencyDetectionItti.h
// Author      : Oytun Akman, oytunakman@gmail.com
// Version     : 1.0
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "A Model of Saliency-Based Visual Attention
//				 for Rapid Scene Analysis" by Laurent Itti, Christof Koch and Ernst
//				 Niebur (PAMI 1998).												  
//===================================================================================

#ifndef _ITTI_H_INCLUDED_
#define _ITTI_H_INCLUDED_

 

// OpenCV
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
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

using namespace cv;
using namespace std;
 

class ITTI
{
 
	 
public:
	ITTI() 
	{
		 
	}


	~ITTI()
	{
		 
	}

	 
	void calculateSaliencyMap(const Mat*  src, Mat * dst, int corlor = 1, int scaleBase = 3);
	 void viewFeaturemaps();
	 void combineFeatureMaps(int corlor,int scale);

	 Mat conspicuityMap_I;
	 Mat conspicuityMap_C;
	 Mat conspicuityMap_O;
	 Mat S;

private:
	 Mat r, g, b, R, G, B, Y, I;
	 vector<Mat> gaussianPyramid_I;
	 vector<Mat> gaussianPyramid_R;
	 vector<Mat> gaussianPyramid_G;
	 vector<Mat> gaussianPyramid_B;
	 vector<Mat> gaussianPyramid_Y;

	 void createChannels(const Mat* src);
	 void createScaleSpace(const Mat* src, vector<Mat>* dst, int scale);

	 void normalize_rgb();
	 void create_RGBY();
	 void createIntensityFeatureMaps();
	 void createColorFeatureMaps();
	 void createOrientationFeatureMaps(int orientation);
	 void mapNormalization(Mat* src);
	 void clearBuffers();

	 vector<Mat> featureMaps_I;
	 vector<Mat> featureMaps_RG;
	 vector<Mat> featureMaps_BY;
	 vector<Mat> featureMaps_0;
	 vector<Mat> featureMaps_45;
	 vector<Mat> featureMaps_90;
	 vector<Mat> featureMaps_135;
};
#endif
