#include "stdafx.h"
#include "detectSaliency.h"

void testSingleImg( string imgPath )
{
	Mat saliencyImg, inputImg8u, savedImg ;
	inputImg8u = imread( imgPath, 1 ) ;
	imshow( "Original Image", inputImg8u ) ;
	clock_t startTime ;

	vector<IParams> iParamSet ;
	//Parameters for PISA
	{
		IParams tmpParam ;
		tmpParam.winLen = 21 ;
		tmpParam.colorTao = 30 ;

		tmpParam.kappa = 0.006f ; 
		tmpParam.lambda = 150.f ;
		tmpParam.borderMargin = 15 ;
		tmpParam.T = 30 ;

		tmpParam.filterRadius = 31 ;
		tmpParam.filterColorTao = 30.f ;
		tmpParam.saliencyLevel = 24 ;

		tmpParam.colorClusterFactor = 1 ;
		tmpParam.omClusterFactor = 1 ;

		iParamSet.push_back( tmpParam ) ;
	}

	//Parameters for F-PISA
	{
		IParams tmpParam ;
		tmpParam.winLen = 11 ;
		tmpParam.colorTao = 50 ;

		tmpParam.kappa = 0.02f ;
		tmpParam.lambda = 70.f ; 
		tmpParam.borderMargin = 5 ;
		tmpParam.T = 30 ;

		tmpParam.filterRadius = 13 ;
		tmpParam.filterColorTao = 30.f ;
		tmpParam.saliencyLevel = 24 ;

		tmpParam.colorClusterFactor = 3 ;
		tmpParam.omClusterFactor = 2 ;

		iParamSet.push_back( tmpParam ) ;
	}

	startTime = clock() ;
	detectSaliencyByPISA( inputImg8u, saliencyImg, iParamSet[0] ) ;
	cout << "Time Used( PISA ): " << ( double )( clock() - startTime ) / CLOCKS_PER_SEC << "s" << endl ;
	namedWindow( "saliency Image( PISA ):" ) ;
	imshow( "saliency Image( PISA ):", saliencyImg ) ;
	saliencyImg.convertTo( savedImg, CV_8UC1, 255 ) ;
	imwrite( "PISA.png", savedImg ) ;

	startTime = clock() ;
	detectSaliencyByFPISA( inputImg8u, saliencyImg, iParamSet[1] ) ;
	cout << "Time Used( F-PISA ): " << ( double )( clock() - startTime ) / CLOCKS_PER_SEC << "s" << endl ;
	namedWindow( "saliency Image( F-PISA ):" ) ;
	imshow( "saliency Image( F-PISA ):", saliencyImg ) ;
	saliencyImg.convertTo( savedImg, CV_8UC1, 255 ) ;
	imwrite( "F-PISA.png", savedImg ) ;

	waitKey( 5000 ) ;
}