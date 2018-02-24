#include "stdafx.h"
#include "detectSaliency.h"

#pragma once ;

/*
	detect salient objects via PISA and F-PISA
*/
void detectSaliency( string imgFolder, string fileNames[], int fileCount, string resultFolder )
{
	vector<IParams> iParamSet ;

	//Parameters for PISA
	{
		IParams tmpParam ;
		tmpParam.winLen = 21 ;
		tmpParam.colorTao = 60 ;

		tmpParam.kappa = 0.006f ; 
		tmpParam.lambda = 150.f ;
		tmpParam.borderMargin = 15 ;
		tmpParam.T = 30 ;

		tmpParam.filterRadius = 31 ;
		tmpParam.filterColorTao = 60.f ;
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

		tmpParam.kappa = 0.035f ;
		tmpParam.lambda = 70.f ; 
		tmpParam.borderMargin = 5 ;
		tmpParam.T = 30 ;

		tmpParam.filterRadius = 11 ;
		tmpParam.filterColorTao = 50.f ;
		tmpParam.saliencyLevel = 24 ;

		tmpParam.colorClusterFactor = 3 ;
		tmpParam.omClusterFactor = 2 ;

		iParamSet.push_back( tmpParam ) ;
	}

	double totalTimeUsed_PISA = 0 ;
	double totalTimeUsed_FPISA = 0 ;

	clock_t startTime = clock() ;

#pragma omp parallel for num_threads( 4 )
	for ( int indx = 0; indx < fileCount; indx++ )
	{
		string imgPath = imgFolder + fileNames[indx] ;
		cout << imgPath << " *index* "<<indx<<endl ;
		Mat img8u = imread( imgPath ) ;		
		{
			Mat sal ;
			clock_t startTime = clock() ;
			detectSaliencyByPISA( img8u, sal, iParamSet[0] ) ;
			totalTimeUsed_PISA += ( double )( clock() - startTime ) / CLOCKS_PER_SEC ;

			string savedPath( resultFolder + fileNames[indx] ) ;
			savedPath = savedPath.substr( 0, savedPath.find_last_of( ".jpg" ) - 3 ) ;
			savedPath += "_PISA" ;
			savedPath +=  ".png" ;
			imwrite( savedPath.c_str(), sal * 255 ) ;
		}
		{
			Mat sal ;
			clock_t startTime = clock() ;
			detectSaliencyByFPISA( img8u, sal, iParamSet[1] ) ;
			totalTimeUsed_FPISA += ( double )( clock() - startTime ) / CLOCKS_PER_SEC ;

			string savedPath( resultFolder + fileNames[indx] ) ;
			savedPath = savedPath.substr( 0, savedPath.find_last_of( ".jpg" ) - 3 ) ;
			savedPath += "_FPISA" ;
			savedPath +=  ".png" ;
			imwrite( savedPath.c_str(), sal * 255 ) ;
		}
	}	
	cout << "Average Time used: " 
		<< ( double )( clock() - startTime ) / CLOCKS_PER_SEC 
		<< "s. (PISA:  " << totalTimeUsed_PISA / fileCount << "s | FPISA: " 
						<< totalTimeUsed_FPISA / fileCount << "s)" << endl ;
}

/*
	Employ PISA and F-PISA to detect saliency objects on given image folder

	Function Name: testImageFolder
	parameters:	
		imgFolder			--		the folder path of original images
		resultFolder		--		the path to save saliency maps
	Implementation detail:
		1£©get all the paths of the imgFolder's images
		2£©detect Saliency object on each given image and save the result
*/
void testImageFolder( string imgFolder, string resultFolder )
{
	string fileNames[10000] ;
	int fileCount = 0 ;
	getFileNames( fileNames, fileCount, imgFolder ) ;

	detectSaliency( imgFolder, fileNames, fileCount, resultFolder ) ;
}
