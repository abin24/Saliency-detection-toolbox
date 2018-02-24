//===================================================================================
// Name        : saliencyDetectionRudinac.h
// Author      : Joris van de Weem, joris.vdweem@gmail.com
// Version     : 1.0
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "Maja Rudinac, Pieter P. Jonker. 
// 				"Saliency Detection and Object Localization in Indoor Environments". 
//				ICPR'2010. pp.404~407											  
//===================================================================================

#ifndef _RUDINAC_H_INCLUDED_
#define _RUDINAC_H_INCLUDED_



// OpenCV
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


class Rudinac
{
 


public:
	Rudinac()
    	{
    		
    	}

	~Rudinac()
    	{
    	
    	}

    	
	void calculateSaliencyMap(const Mat* src, Mat* dst, int corlor = 1);

private:
    	Mat r,g,b,RG,BY,I;
    	void createChannels(const Mat* src);
    	void createSaliencyMap(const Mat src, Mat* dst);
};
#endif
