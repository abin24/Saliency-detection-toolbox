#include "stdafx.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <ctime>
#include "CLMF_SDK_Import.h"  

#pragma once


void performCostVolumeFiltering( Mat& dstSal, const Mat& srcSal, const Mat& img8u, const int LabelNum = 24, int filterOrder = 0, int radius = 15, float colorTau = 50, float epsl = 0.01f )
{
	int m = srcSal.rows ;
	int n = srcSal.cols ;
	int maxLabel = LabelNum - 1 ;
	vector<Mat> dispVol( LabelNum ) ;

//#pragma omp parallel for num_threads( 4 )
	for ( int i = 0; i < LabelNum; i++ )
	{
		//construct the cost-volume
		dispVol[i] = Mat::zeros( m, n, CV_32FC1 ) ;
		Mat tmp = srcSal * maxLabel - (double)i ;		
		dispVol[i] = tmp.mul( tmp ) ;

		//filter the cost-volume
		FilterImageWithCLMF( 2, filterOrder, img8u, dispVol[i], dispVol[i], radius, colorTau, epsl, 1 ) ;
	} 
	
	//select label( Winner take all label selection )
	dstSal = Mat::zeros( img8u.size(), CV_32FC1 ) ;
	const float** dataPtr = new const float*[LabelNum] ;

	for( size_t row = 0; row < img8u.rows; row++ )
	{
		float* salPtr = dstSal.ptr<float>( ( int )row ) ;
		
		for( size_t i = 0; i < LabelNum; i++ )
			dataPtr[i] = dispVol[i].ptr<float>( ( int )row ) ;

		for( size_t col = 0; col < img8u.cols; col++ )
		{
			float minVal = dataPtr[0][col] ;
			for( size_t i = 1; i < LabelNum; i++ )
			{
				if ( minVal > dataPtr[i][col] )
				{
					salPtr[col] = ( float )i ;
					minVal = dataPtr[i][col] ;
				}
			}
		}
	}
	delete [] dataPtr ;

	normalize( dstSal, dstSal, 0, 1, NORM_MINMAX ) ;
}