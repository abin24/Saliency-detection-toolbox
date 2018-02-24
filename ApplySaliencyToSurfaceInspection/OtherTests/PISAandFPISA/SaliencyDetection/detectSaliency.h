#include "stdafx.h"
#include "ColorBasedContrast.h"
#include "StructureBasedContrast.h"
#include "filterImg.h"
#include "performCostVolumeFiltering.h"

#pragma once ;

/*
	PISA: Detect Image Saliency based on color and structure contrast with spatial priors.
	\hat{f}( p ) = Uc( p ) * Dc( p ) + Ug( p ) * Dg( p )
	E = \Arrowvert S( p ) - f(p) \Arrowvert _2^2 + \sum_{q \in \Omega_p}{\omega_{pq} \Arrowvert S_p-S_q \Arrowvert _2^2},
	Employ the fast cost-volume filtering technique to solve E.
	
	Parameters:
		Input:
			img8u(CV_8UC3): input image 
			params(IParams): parameters
		Output:
			sal(CV_32FC1): the detected saliency of the input image
*/
void detectSaliencyByPISA( const Mat& img8u, Mat& sal, IParams params ) 
{
	Mat img32f ;
	img8u.convertTo( img32f, CV_32FC1, 1 / 255.0 ) ;

	Mat strucSal, strucClusterIndx, strucClusterRank, strucClusterVariance ;
	Mat colorSal, colorClusterIndx, colorClusterRank, colorClusterVariance ;

	calcSaliencyByOM( strucSal, strucClusterIndx, strucClusterRank, strucClusterVariance, img8u, img32f, params ) ;
	calcSaliencyByColor( colorSal, colorClusterIndx, colorClusterRank, colorClusterVariance, img8u, img32f, params ) ;	
	
	descideSaliency( sal, 
		strucSal, colorSal, 
		strucClusterIndx, colorClusterIndx, 
		strucClusterRank, colorClusterRank, 
		strucClusterVariance, colorClusterVariance, params.T ) ;	

	performCostVolumeFiltering( sal, sal.clone(), img8u, params.saliencyLevel, 0, params.filterRadius, params.filterColorTao, 0.01f ) ;
	normalizeBySigmoid( sal.clone(), sal ) ;
}

/*
	F-PISA: Perform a gradient-driven subsampling of the input image and Detect Image Saliency based on color and structure contrast with spatial priors.

	\hat{f}( p ) = Uc( p ) * Dc( p ) + Ug( p ) * Dg( p )
	E = \Arrowvert S( p ) - f(p) \Arrowvert _2^2 + \sum_{q \in \Omega_p}{\omega_{pq} \Arrowvert S_p-S_q \Arrowvert _2^2}.
	Employ the fast cost-volume filtering technique to solve E.
	Parameters:
		Input:
			img8u(CV_8UC3): input image 
			params(IParams): parameters
		Output:
			sal(CV_32FC1): the detected saliency of the input image
*/
void detectSaliencyByFPISA( const Mat& img8u, Mat& sal, IParams params ) 
{
	Mat downSmpImg ;
	getSelectedPoints( downSmpImg, img8u, Size( params.winLen, params.winLen ) ) ;

	Mat img32f ;
	downSmpImg.convertTo( img32f, CV_32FC1, 1 / 255.0 ) ;

	Mat strucSal, strucClusterIndx, strucClusterRank, strucClusterVariance ;
	Mat colorSal, colorClusterIndx, colorClusterRank, colorClusterVariance ;

	calcSaliencyByOM( strucSal, strucClusterIndx, strucClusterRank, strucClusterVariance, downSmpImg, img32f, params ) ;
	calcSaliencyByColor( colorSal, colorClusterIndx, colorClusterRank, colorClusterVariance, downSmpImg, img32f, params ) ;		
	
	descideSaliency( sal, 
		strucSal, colorSal, 
		strucClusterIndx, colorClusterIndx, 
		strucClusterRank, colorClusterRank, 
		strucClusterVariance, colorClusterVariance, params.T ) ;	
	
	jointlyUpsampleByCLMF( img8u, sal, sal.clone(), params.filterRadius, params.filterColorTao ) ;
	normalizeBySigmoid( sal.clone(), sal ) ;
}