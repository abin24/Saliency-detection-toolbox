//===================================================================================
// Name        : saliencyDetectionHou.h
// Author      : Oytun Akman, oytunakman@gmail.com
// Version     : 1.0
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "Saliency Detection: A Spectral Residual 
//				 Approach" by Xiaodi Hou and Liqing Zhang (CVPR 2007).												  
//===================================================================================

#ifndef _SALIENCYMAPHOU_H_INCLUDED_
#define _SALIENCYMAPHOU_H_INCLUDED_

// ROS
 

// OpenCV
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
 

class SR
{
 
     

public:
    	SR()  
    	{
    		 
    	}


    	~SR()
    	{
    		 
    	}

    	//void imageCB(const sensor_msgs::ImageConstPtr& msg_ptr);
    	void calculateSaliencyMap(const Mat* src, Mat* dst);
};
#endif
