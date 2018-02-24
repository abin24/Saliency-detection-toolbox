#include "stdafx.h"
#include "clusterHistByKMeansPPModified.h"
#include "SpatialPrior.h"

#include "stdafx.h"

#pragma once

/*
	Linearly-varying Smoothing Scheme
	Parameters:
		Input:
			clusterHist: histogram of each cluster
			similar: similar clusters
			delta: size of smoothing
		Output:
			binSal: saliency value of each cluster
*/
void SmoothByLinearlyVarying( Mat& dstBinSal, const Mat &clusterHist, const Mat& srcBinSal, float watchWndFactor, const Mat& distMat )
{
	assert( clusterHist.cols > 2 ) ;

	int binN = clusterHist.rows ;
	int watchWindow = max( cvRound( binN / watchWndFactor ), 2 ) ;

	Mat sortedMat, sortedIdx ;
	cv::sort( distMat, sortedMat, CV_SORT_ASCENDING + CV_SORT_EVERY_ROW ) ;
	sortIdx( distMat, sortedIdx, CV_SORT_ASCENDING + CV_SORT_EVERY_ROW ) ;

	const float* srcSalPtr = srcBinSal.ptr<float>( 0 ) ;
	dstBinSal = Mat::zeros( 1, binN, CV_32FC1 ) ;
	float* dstSalPtr = dstBinSal.ptr<float>( 0 ) ;
	for( int i = 0; i < binN; i++ )
	{
		const float* sortedMatPtr = sortedMat.ptr<float>( i ) ;
		const int* sortedIdxPtr = sortedIdx.ptr<int>( i ) ;
		float totalDist = 0 ;

		for( int j = 1; j < watchWindow; j++ )
		{
			totalDist += sortedMatPtr[j] ;
		}

		for ( int j = 0; j < watchWindow; j++ )
		{
			dstSalPtr[i] += srcSalPtr[sortedIdxPtr[j]] * ( totalDist - sortedMatPtr[j] ) ; 
		}
		dstSalPtr[i] /= ( (watchWindow - 1) * totalDist ) ;
	}
}

/*
	Use the rarity of clusters as the proxy to evaluate the rarity or contrast measure for pixels, which is same as HC
	Parameters:
		Input:
			clusterHist: the histogram of each cluster 
			histWeight: the frequency of each cluster
			idx: index map of every pixel
		Output:
			imgSal: image saliency map of every pixel
*/


/*
	Use the rarity of clusters as the proxy to evaluate the rarity or contrast measure for pixels, which is same as Cheng ming ming's HC
	Parameters:
		Input:
			clusterHist: the histogram of each cluster 
			histWeight: the frequency of each cluster
			idx: index map of every pixel
		Output:
			imgSal: image saliency map of every pixel
*/
void calcClusterContrast( Mat& imgSal, const Mat& idx, const Mat& clusterHist, const Mat& histWeight )
{
	const int binNum = clusterHist.rows ;
	const int dimNum = clusterHist.cols ;
	Mat binSal = Mat::zeros( 1, binNum, CV_32FC1 ) ;
	Mat distMat = Mat::zeros( binNum, binNum, CV_32FC1 ) ;
	const float* weightPtr = histWeight.ptr<float>( 0 ) ;
	float* binSalPtr = binSal.ptr<float>( 0 ) ;

	for ( int i = 0; i < binNum; i++ )
	{
		float* distMatPtr = distMat.ptr<float>( i ) ;
		const float* hist_i = clusterHist.ptr<float>( i ) ;

		for ( int j = 0; j < binNum; j++ )
		{
			if ( i == j )
				continue ;
			
			const float* hist_j = clusterHist.ptr<float>( j ) ;
			float dist_ij = normL2Sqr_( hist_i, hist_j, dimNum ) ;
			distMatPtr[j] = dist_ij ;
			binSalPtr[i] += weightPtr[j] * dist_ij ; 
		}
	}
	SmoothByLinearlyVarying( binSal, clusterHist, binSal.clone(), 4.0f, distMat ) ;

	int rows = idx.rows ;
	int cols = idx.cols ;
	imgSal = Mat::zeros( rows, cols, CV_32FC1 ) ;

	for( int row = 0; row < rows; row++ )
	{
		float* salPtr = imgSal.ptr<float>( row ) ;
		const int* idxPtr = idx.ptr<int>( row ) ;

		for ( int col = 0; col < cols; col++ )
		{
			salPtr[col] = binSalPtr[idxPtr[col]] ;
		}
	}
}

/*
Calculate U * D for color-based or structure-based contrast.
Parameters:
	Input: 
		img32f( CV_32FC3 ): input image
		response( CV_32FC1 ): feature vector for image pixels ( For N pixels, response has N rows )
		centerNum: the number of K in kmeans clustering.
		params: parameters
	Output: 
		imgSal( CV_32FC1 ): image saliency detection result
		idx( CV_32SC1 ): index map for every pixel of clusters
		rank: rank map for every pixel of clusters
		variance: variance map for every pixel of clusters 
Implement Details:
	1. Cluster response histograms by using modified kmeans++
	2. Calculate contrast term U according to the rarity of clusters
	3. Calculate spatial prior term D
	4. calcuate U * D
*/
void calcClusterSaliencyWithSpatialPriors( Mat& imgSal, Mat& idx, Mat& rank, Mat& variance, const Mat& response, int centerNum, const Mat& img32f, const IParams& params )
{
	Mat clusterHist, clusterWeight ;
	clusterHistByKMeansPPModified( idx, clusterHist, clusterWeight, response, centerNum, img32f ) ; 	
	idx = idx.reshape( 0, img32f.rows ) ;

	calcClusterContrast( imgSal, idx, clusterHist, clusterWeight ) ;

	Mat centerSal ;
	calcCenterPreference( centerSal, rank, variance, idx, clusterHist, clusterWeight, params.kappa ) ;	

	Mat borderSal ;
	calcBorderExclusion( borderSal, idx, clusterHist.rows, params.lambda, params.borderMargin );

	imgSal = imgSal.mul( centerSal ).mul( borderSal ) ;

	normalize( imgSal, imgSal, 0, 1, NORM_MINMAX ) ;
}