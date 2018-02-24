#include "stdafx.h"

#pragma once

/*
	constrain pixels rendered salient to be centered in image domain by computing distance to the image center and image border.
	Parameters:
		Input:
			idx: index map of every pixel
			clusterHist: histogram of each cluster
			histWeight: frequency of each cluster
			k: which controls the fall-off rate of the exponential function, default is 0.01, but for better result, you should take parameters.txt file into consideration.  
		Output:
			spatialSal: final spatial term
			clusterVarianceRank: rank map of variance of every pixel
			clusterVariance: variance map of every pixel
		Implement Details
			1) calculate the distance of each cluster to the image center
			2) weight the cluster by their distances.
*/
void calcCenterPreference( Mat& spatialSal, Mat& clusterVarianceRank, Mat& clusterVariance, const Mat& idx, const Mat& clusterHist, const Mat& histWeight, float kappa )
{
	int binNum = clusterHist.rows ;
	Mat binSal = Mat::zeros( 1, binNum, CV_32FC1 ) ;
	float* binSalPtr =( float* )( binSal.data ) ;
	float *weight = ( float* )( histWeight.data ) ;

	vector<Point> ClusterCenter(binNum);
	vector<int> PixNum(binNum);
	for( int row = 0; row < idx.rows; row++ )
	{
		const int* idxPtr = idx.ptr<int>( row ) ;
		for ( int col = 0; col < idx.cols; col++ )
		{
			int tempIdx = idxPtr[col] ;
			PixNum[tempIdx] ++;
			ClusterCenter[tempIdx].x += col;
			ClusterCenter[tempIdx].y += row;
		}
	}
	for(int i = 0; i < binNum; i++)
	{
		if ( PixNum[i] != 0 )
		{
			ClusterCenter[i].x /= PixNum[i];
			ClusterCenter[i].y /= PixNum[i];
		}
	}

	vector<float> ClusterVariance(binNum);
	vector<float> ClusterVariance_center(binNum);
	for( int row = 0; row < idx.rows; row++ )
	{
		const int* idxPtr = idx.ptr<int>( row ) ;
		for ( int col = 0; col < idx.cols; col++ )
		{
			int tempIdx = idxPtr[col] ;
			float disTemp = sqrt((float)(row - idx.rows / 2) * (row - idx.rows / 2)
					+ (float)(col - idx.cols / 2) * (col - idx.cols / 2));
			ClusterVariance_center[tempIdx] += disTemp;
		}
	}
	for(int i = 0; i < binNum; i++)
	{
		if ( PixNum[i] != 0 )
		{
			ClusterVariance_center[i] /= (float)PixNum[i];
			ClusterVariance[i] = ClusterVariance_center[i] * ClusterVariance_center[i];
		}
	}

	for( int i = 0; i < binNum; i++ )
	{
		binSalPtr[i] = exp ( -kappa * kappa * ClusterVariance[i] );
	}

	spatialSal = Mat::zeros( idx.size(), CV_32FC1 ) ;
	for( int row = 0; row < idx.rows; row++ )
	{
		float* salPtr = spatialSal.ptr<float>( row ) ;
		const int* idxPtr = idx.ptr<int>( row ) ;
		for ( int col = 0; col < idx.cols; col++ )
		{
			salPtr[col] = binSalPtr[idxPtr[col]] ;
		}
	}

	vector<float> tmpClusterVariance ( ClusterVariance ) ;
	sort( tmpClusterVariance.begin(), tmpClusterVariance.end() ) ;

	clusterVarianceRank = Mat::zeros( 1, binNum, CV_32FC1 ) ;
	clusterVariance = Mat::zeros( 1, binNum, CV_32FC1 ) ;
	float* rankPtr = clusterVarianceRank.ptr<float>( 0 ) ;
	float* variancePtr = clusterVariance.ptr<float>( 0 ) ; 
	for( int i = 0; i < binNum; i++ )
	{
		for( int j = 0; j < binNum; j++ )
		{
			if ( ClusterVariance[i] == tmpClusterVariance[j] )
			{
				rankPtr[i] = (float)j ;
			}
		}
		variancePtr[i] = ClusterVariance[i] ;
	}
}

/*
	calculate the probability map of each cluster belong to the image border region
	Parameters:
		Input:
			idx: index map of every pixel
			binNum: number of clusters
			sigma: weight parameter between image boundary exclusion and center preference  
		    borderMargin: pixels number close to image boundary, control the size of border region
		Output:
			spatialSal: final boundary exclusion map	
*/
void calcBorderExclusion(Mat& spatialSal, const Mat& idx, int binNum, float lambda, int borderMargin )
{
	int width = idx.cols, height = idx.rows;
	int borderPixNum = borderMargin * 2 * (width + height) - 60;
	float* clusterPixInBorder = new float[binNum];
	memset(clusterPixInBorder, 0, binNum * sizeof(float));

	for(int i = 0; i < borderMargin; i++)
	{
		const int* idxPtr = idx.ptr<int>( i ) ;

		for( int j = 0; j < width; j++)
			clusterPixInBorder[idxPtr[j]] += 2;
	}

	for( int i = height - borderMargin; i < height; i++ )
	{
		const int* idxPtr = idx.ptr<int>( i ) ;

		for( int j = 0; j < width; j++)
			clusterPixInBorder[idxPtr[j]] += 0.4f;
	}

	for(int i = borderMargin; i < height - borderMargin; i++)
	{
		const int* idxPtr = idx.ptr<int>( i ) ;

		for ( int j = 0; j < borderMargin; j++ )
			clusterPixInBorder[idxPtr[j]] += 0.8f;

		for(int j = width - borderMargin; j < width;j++)
			clusterPixInBorder[idxPtr[j]] += 0.8f;
	}


	for( int i = 0; i < binNum; i++ )
	{
		clusterPixInBorder[i] = exp ( -lambda * clusterPixInBorder[i] /  borderPixNum );
	}

	spatialSal = Mat::zeros( height, width, CV_32FC1 ) ;
	for(int i = 0;i < height;i++)
	{
		float* imgSalPtr = spatialSal.ptr<float>(i);
		const int* idxPtr = idx.ptr<int>( i ) ;
		for(int j = 0;j < width;j++)
		{
			imgSalPtr[j] = clusterPixInBorder[idxPtr[j]];
		}
	}
	normalize( spatialSal, spatialSal, 0, 1, NORM_MINMAX ) ;
}