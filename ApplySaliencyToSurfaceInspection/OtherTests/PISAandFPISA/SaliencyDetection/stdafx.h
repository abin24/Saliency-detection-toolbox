#define _CRT_SECURE_NO_WARNINGS
#pragma once

#include <iostream>
#include <map>
#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <io.h>
#include <omp.h>
#include <list>
#include <string>
#include <fstream>

#pragma once

using namespace std ;
using namespace cv ;

void getFileNames( string fileNames[], int& fileCount, string imgFolder, string subFolderPath = "" ) ;
void normalizeBySigmoid( const Mat& srcSal, Mat& dstSal, int halfRange = 5 ) ;
void descideSaliency( Mat& sal, 
	const Mat& strucSal, const Mat& colorSal, 
	const Mat& strucClusterIndx, const Mat& colorClusterIndx, 
	const Mat& strucClusterRank, const Mat& colorClusterRank, 
	const Mat& strucClusterVar, const Mat& colorClusterVar, 
	int thresh
	) ;
int getFrequentFeatureNum( map<int, int>& pallet, int pixelNum, float ratio ) ;

struct IParams
{
	//Parameters for feature extraction
	int winLen ; 
	float colorTao ;

	float kappa ;
	float lambda ;
	int borderMargin ;

	//Parameters for filtering
	int filterRadius ;
	float filterColorTao ;
	
	int T ;
	int saliencyLevel ;

	int omClusterFactor ;
	int colorClusterFactor ;
};

//Parameter for subsampling
const int stepSize = 3 ;

//Parameter for orientation and magnitude (OM) feature
const int OrientationBinNum = 8 ;
const int MagnitudeBinNum = 8 ;
const int ColorChannelBinNum = 12 ;

//Help function
template<typename T> inline T sqr(T x) { return x * x;}
template<class T> inline T calcVec3fDist(const Vec<T, 3> &v1, const Vec<T, 3> &v2) {return (T)sqrt( 0.f + ( sqr(v1[0] - v2[0])+sqr(v1[1] - v2[1])+sqr(v1[2] - v2[2])) );}