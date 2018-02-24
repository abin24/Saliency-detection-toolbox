#include "stdafx.h"
#include "CLMF_SDK_Import.h"
#include "calcClusterSaliencyWithSpatialPriors.h"

#pragma once ;

/*
	We construct an shape-adaptive observation region efficiently for every pixel in the image using CLMF, and then we calculate color histogram(36 bins) for every pixel in the image by using soft quantization
	Parameters:
		Input:
			img8u: original image
			maxArm: the maximum arm length, ie. L in the paper
			img32s(CV_32SC3): Lab image of the original image
		Output:
			colorResponse: store color histogram for every pixel in the image
*/
void getColorResponse( Mat& response, const Mat& img8u, const Mat& img32s, int maxArm, float threshold )
{
	cv::Mat_<cv::Vec3b> img( img8u ) ;
	Mat_<Vec3b> cmUD, cmLR ;
	SaveOutCrossSupportMap( img, cmUD, cmLR, maxArm, threshold, 1 );
	
	const float preRes = 11 / 255.f ;
	const int rows = img32s.rows ;
	const int cols = img32s.cols ;
	const int dim = ColorChannelBinNum * 3 ;
	const int imgSize = rows * cols ;

	response = Mat::zeros( imgSize, dim, CV_32FC1 ) ;

	assert( response.isContinuous() ) ;

	Mat histCache = Mat::zeros( imgSize, dim, CV_32FC1 ) ;
	int* pixelColCache = new int[imgSize] ;
	
	memset( pixelColCache, 0, sizeof( int ) * imgSize ) ;
	for ( int row = 0; row < rows; row++ )
	{
		const Vec3b* lrPtr = cmLR.ptr<Vec3b>( row ) ;
		for ( int col = 0; col < cols; col++ )
		{
			int indx = row * cols + col ;
			float* cachePtr = histCache.ptr<float>( indx ) ;
			
			const Vec3i* colorPtr = img32s.ptr<Vec3i>( row ) ;
			for ( int x = col - lrPtr[col][0]; x <= col + lrPtr[col][1]; x++ )
			{
				for( int i = 0; i < 3; i++ )
				{
					float val = colorPtr[x][i] * preRes ;
					int downBin = (int)val ;
					float res = downBin - val ;
					downBin += i * ColorChannelBinNum ;
					cachePtr[downBin] += res + 1 ;
					cachePtr[downBin + 1] += -res ;			
				}
			}
			pixelColCache[indx] = lrPtr[col][0] + lrPtr[col][1] + 1 ;
		}
	}

	__m128 a, b, c ;
	
	for ( int row = 0; row < rows; row++ )
	{
		const Vec3b* udPtr = cmUD.ptr<Vec3b>( row ) ;
		for ( int col = 0; col < cols; col++ )
		{
			int indx = row * cols + col ;
			float* responsePtr = response.ptr<float>( indx ) ;
			float pixelNum = 0 ;
			for ( int y = row - udPtr[col][1]; y <= row + udPtr[col][2]; y++ )
			{
				int cacheIdx = y * cols + col ;
				const float* cachePtr = histCache.ptr<float>( cacheIdx ) ; 		
				
				for( int i = 0; i < dim; i += 4 )
				{
					a = _mm_load_ps( responsePtr + i ) ;
					b = _mm_load_ps( cachePtr + i ) ;
					c = _mm_add_ps( a, b ) ;
					_mm_store_ps( responsePtr + i, c ) ;
				}

				pixelNum += pixelColCache[cacheIdx] ;
			}
           
			if ( 0 < pixelNum )
			{
				b = _mm_setr_ps( pixelNum, pixelNum, pixelNum, pixelNum ) ;
				for ( int i = 0; i < dim; i += 4 )
				{
					a = _mm_load_ps( responsePtr + i ) ;
					c = _mm_div_ps( a, b ) ;
					_mm_store_ps( responsePtr + i, c ) ;
				}
			}
		}
	}

	delete [] pixelColCache ;
}


/* Decide cluster number K by choosing most frequently occurring features by ensuring they cover XX% of histogram distributions,
and the feature is defined by the combination of two histograms.
	Parameters:
		Input:
			img32f (CV_32FC3): input image
			dimNum: quantized color channel dimension, here is 12.
			ratio: XX%, default is 95%
		Output:
			return cluster number K
*/ 
int getClusterNumByQuantize4Color( const Mat& img32f, int dimNum, float ratio )
{
	Mat tmp[3] ;
	split( img32f, tmp ) ;

	const int rows = tmp[0].rows ;
	const int cols = tmp[0].cols ;
	const float dimNumf = dimNum - 0.0001f ;

	map<int, int> pallet ;
	int idx ;
	for ( int row = 0; row < rows; row++ )
	{
		const float* aPtr = tmp[0].ptr<float>( row ) ;
		const float* bPtr = tmp[1].ptr<float>( row ) ;
		const float* cPtr = tmp[2].ptr<float>( row ) ;
		for( int col = 0; col < cols; col++ )
		{
			idx = ( int )( aPtr[col] * dimNumf ) * dimNum * dimNum + ( int ) ( bPtr[col] * dimNumf ) * dimNum + (int)( cPtr[col] * dimNumf ) ;
			pallet[idx]++ ;
		}
	}

	return getFrequentFeatureNum( pallet, rows * cols, ratio ) ;
}

void calcSaliencyByColor( Mat& imgSal, Mat& idx, Mat& rank, Mat& variance, const Mat& img8u, const Mat& img32f, const IParams& params )
{
	Mat imgLAB8u, imgLAB32s ;

	cvtColor( img8u, imgLAB8u, CV_RGB2Lab ) ;
	imgLAB8u.convertTo( imgLAB32s, CV_32SC3 ) ;

	int colorCenterNum = getClusterNumByQuantize4Color( img32f, ColorChannelBinNum, 0.95f ) / params.colorClusterFactor ;

	Mat colorResponse ;
	getColorResponse( colorResponse, img8u, imgLAB32s, params.winLen, params.colorTao ) ;

    calcClusterSaliencyWithSpatialPriors( imgSal, idx, rank, variance, colorResponse, colorCenterNum, img32f, params ) ;
}
