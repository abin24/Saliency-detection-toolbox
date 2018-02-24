#include "stdafx.h"
#include "clusterHistByKMeansPPModified.h"
#include "SpatialPrior.h"

#pragma once ;

/* Decide cluster number K by choosing most frequently occurring features by ensuring they cover XX% of histogram distributions,
and the feature is defined by the combination of two histograms.
	Parameters:
		Input:
			histA( Vec2b ): orientation histogram
			histB( CV_32FC1 ): magnitude histogram
			ratio: XX%, default is 95%
		Output:
			return cluster number K
*/  
int getClusterNumByQuantize4Struct( const Mat& histA, const Mat& histB, int dimNum, float ratio = 0.95f )
{
	Mat tmp[2] ;
	split( histA, tmp ) ;
	Mat tmpA ;
	//choose first orientation
	tmp[0].convertTo( tmpA, CV_32SC1, 1 ) ;

	Mat tmpB = histB + 0.5 ;
	tmpB.convertTo( tmpB, CV_32SC1, 1 ) ;

	const int rows = tmpA.rows ;
	const int cols = tmpA.cols ;

	// build pallet
	Mat idx = Mat::zeros( rows, cols, CV_32SC1 ) ;
	map<int, int> pallet ;
	for ( int row = 0; row < rows; row++ )
	{
		const int* aPtr = tmpA.ptr<int>( row ) ;
		const int* bPtr = tmpB.ptr<int>( row ) ;
		int* idxPtr = idx.ptr<int>( row ) ;
		for( int col = 0; col < cols; col++ )
		{
			idxPtr[col] = aPtr[col] + bPtr[col] * dimNum ;
			pallet[idxPtr[col]]++ ;
		}
	}

	return getFrequentFeatureNum( pallet, rows * cols, ratio ) ;
}

/*
	calculate gradient map and orientation map of input image by using the API of HOGDescriptor provided by OpenCV
	Parameters:
		Input: 
			img8u(CV_8UC3): input color image
			winSize: size of HOGDescriptor
		Output:
			magQuantized(CV_32FC1): the maximum-quantized magnitude of gradient map of the input image
			grad(CV_32FC2): the gradient map of the input image
			orientation(CV_32FC2): the orientation map of the input image
			resizedImg8u(CV_8UC3): border handled resized image
*/
void getGradFeatures( Mat& magQuantized, Mat& grad, Mat& orientation, Mat& resizedImg8u, const Mat& img8u, Size winSize )
{
	//handle the border
	int winLen = winSize.width / 2 ;
	resizedImg8u = Mat::zeros( img8u.rows + winLen * 2, img8u.cols + winLen * 2, CV_8UC3 ) ;
	Rect roi( winLen, winLen, img8u.cols, img8u.rows ) ;
	img8u.copyTo( resizedImg8u( roi ) ) ;

	Mat srcUpROI = img8u( Rect( 0, 0, img8u.cols, winLen ) ).clone() ;
	flip( srcUpROI, srcUpROI, 0 ) ;
	srcUpROI.copyTo( resizedImg8u( Rect( winLen, 0, img8u.cols, winLen ) ) ) ;

	Mat srcDownROI = img8u( Rect( 0, img8u.rows - winLen, img8u.cols, winLen ) ).clone() ;
	flip( srcDownROI, srcDownROI, 0 ) ;
	srcDownROI.copyTo( resizedImg8u( Rect( winLen, resizedImg8u.rows - winLen, img8u.cols, winLen ) ) ) ;

	Mat srcLeftROI = resizedImg8u( Rect( winLen, 0, winLen, resizedImg8u.rows ) ).clone() ;
	flip( srcLeftROI, srcLeftROI, 1 ) ;
	srcLeftROI.copyTo( resizedImg8u( Rect( 0, 0, winLen, resizedImg8u.rows ) ) ) ;

	Mat srcRightROI = resizedImg8u( Rect( resizedImg8u.cols - 2 * winLen, 0, winLen, resizedImg8u.rows ) ).clone() ;
	flip( srcRightROI, srcRightROI, 1 ) ;
	srcRightROI.copyTo( resizedImg8u( Rect( resizedImg8u.cols - winLen, 0, winLen, resizedImg8u.rows ) ) ) ;

	HOGDescriptor hog( winSize, winSize, winSize, winSize, OrientationBinNum ) ;
	hog.computeGradient( resizedImg8u, grad, orientation ) ;

	Mat gradXY[2] ;
	split( grad, gradXY ) ;
	Mat mag( gradXY[0] + gradXY[1] ) ;
	
	sqrt( mag, mag ) ; 

	normalize( mag, mag, 0, 1, CV_MINMAX ) ;
	mag.convertTo( magQuantized, CV_32FC1, ( MagnitudeBinNum - 1 ) ) ;
}

/*
	calculate Orientation-Magnitude( OM ) histogram (16 bins) for every pixel in the image by using soft quantization
	Parameters:
		Input:
			orient: orientation map of the original image
			grad: gradient map of the original image
			mag: magnitude map of the original image
			winLen: size of local window
		Output:
			response: store OM histogram for every pixel in the image
*/
void getOMResponse( Mat& response, const Mat& orient, const Mat& grad, const Mat& mag, int winLen )
{
	const int rows = orient.rows ;
	const int cols = orient.cols ;
	const int dim = OrientationBinNum + MagnitudeBinNum ;
	  
	response = Mat::zeros( ( rows - winLen / 2 * 2 ) * ( cols - winLen / 2 * 2 ), dim, CV_32FC1 ) ;
	Mat histCache = Mat::zeros( rows * ( cols - winLen / 2 * 2 ), dim, CV_32FC1 ) ;

	assert( response.isContinuous() ) ;

	int cacheIdx = 0 ;
	for( int row = 0; row < rows; row++ )
	{
		for( int col = winLen / 2; col < cols - winLen / 2; col++ )
		{
			float* cachePtr = histCache.ptr<float>( cacheIdx++ ) ;

			const float* magPtr = mag.ptr<float>( row ) ;
			const Vec2b* orientPtr = orient.ptr<Vec2b>( row ) ;
			const Vec2f* gradPtr = grad.ptr<Vec2f>( row ) ;
			for ( int x = col - winLen / 2; x <= col + winLen / 2; x++ )
			{
				float gradPtr1 = gradPtr[x][0] ;
				float gradPtr2 = gradPtr[x][1] ;
				float sumGradPtr = gradPtr1 + gradPtr2 ;
				if ( sumGradPtr != 0 )
				{
					//soft quantization
					cachePtr[(int)( orientPtr[x][0] )] += gradPtr1 / sumGradPtr ;
					cachePtr[(int)( orientPtr[x][1] )] += gradPtr2 / sumGradPtr ;
				}

				int val = ( int )magPtr[x] ;
				float res = magPtr[x] - val;
				cachePtr[OrientationBinNum + val] += 1 - res ;
				val =  ( val + 1 ) < MagnitudeBinNum ? val + 1 : 0 ; 
				cachePtr[OrientationBinNum + val] += res ;										
			}
		}
	}

	__m128 a, b, c ;

	int indx = 0 ;
	for( int row = winLen / 2; row < rows - winLen / 2; row++ )
	{
		for( int col = winLen / 2; col < cols - winLen / 2; col++ )
		{
			float* responsePtr = response.ptr<float>( indx++ ) ;

			for( int y = row - winLen / 2; y <= row + winLen / 2; y++ )
			{
				int cacheIdx =  y * ( cols - winLen / 2 * 2 )  + col - winLen / 2 ;
				const float* cachePtr = histCache.ptr<float>( cacheIdx ) ;
				
				for( int i = 0; i < dim; i += 4 )
				{
					a = _mm_load_ps( responsePtr + i ) ;
					b = _mm_load_ps( cachePtr + i ) ;
					c = _mm_add_ps( a, b ) ;
					_mm_store_ps( responsePtr + i, c ) ;
				}
			}
		}
	}

	response /= winLen * winLen ;
}


/*
	selected pixels with the largest gradient magnitude from a 3 * 3 rectangular patch on the regular image grid
*/
void selectPoints( Point* points, const Mat& mag, int downSmpRows, int downSmpCols, int winLen )
{
	for( int row = winLen / 2 + stepSize / 2; row  < downSmpRows * stepSize + winLen / 2; row += stepSize )
	{
		for( int col = winLen / 2 + stepSize / 2; col < downSmpCols * stepSize + winLen / 2; col += stepSize )
		{
			float maxMag = mag.at<float>( row, col ) ;
			Point maxMagPos = Point( col, row ) ;
			for( int y = row - stepSize / 2; y <= row + stepSize / 2; y++ )
			{
				const float* magPtr = mag.ptr<float>( y ) ;
				for ( int x = col - stepSize / 2; x <= col + stepSize / 2; x++ )
				{
					if( maxMag < magPtr[x] )
					{
						maxMag = magPtr[x] ;
						maxMagPos = Point( x, y ) ;
					}
				}
			}
			points[ ( row - ( winLen / 2 + stepSize / 2 ) ) / stepSize * downSmpCols + ( col - ( winLen / 2 + stepSize / 2 ) ) / stepSize ] 
			= Point( maxMagPos.x - winLen / 2, maxMagPos.y - winLen / 2 ) ;
		}
	}
}

/*
	get selected pixels with the largest gradient magnitude from a 3 * 3 rectangular patch on the regular image grid
*/
void getSelectedPoints( Mat& downSmpImg, const Mat& img8u, Size winSize )
{
	Mat mag, grad, orientation, resizedImg8u ;
	getGradFeatures( mag, grad, orientation, resizedImg8u, img8u, winSize ) ;

	int downSmpRows = cvFloor( img8u.rows * 1.f / stepSize ) ;
	int downSmpCols = cvFloor( img8u.cols * 1.f / stepSize ) ;
	Point *pointsHold = new Point[downSmpRows * downSmpCols] ;
	
	selectPoints( pointsHold, mag, downSmpRows, downSmpCols, winSize.width ) ;
	downSmpImg = Mat::zeros( downSmpRows, downSmpCols, CV_8UC3 ) ;
	for( int row = 0, count = 0; row < downSmpRows; row++ )
	{
		Vec3b* dataPtr = downSmpImg.ptr<Vec3b>( row ) ;
		for( int col = 0; col < downSmpCols; col++, count++ )
		{
			int y = pointsHold[count].y ;
			int x = pointsHold[count].x ;
			dataPtr[col] = img8u.at<Vec3b>( y, x ) ;
		}
	}
	delete[] pointsHold ;
}

/*
	calculate Ug * Dg by using OM ( Orientation Magnitude Histogram ) in structure contrast measure
	Parameters:
		Input: 
			img8u( CV_8UC3 ): input image
			params: parameters
		Output: 
			imgSal( CV_32FC1 ): image saliency detection result
			idx( CV_32SC1 ): index map for every pixel of clusters
			rank: rank map for every pixel of clusters
			variance: variance map for every pixel of clusters 
		Implement Details:
			1. Extract gradient magnitude and gradient orientation with the method provided by OpenCV.
			2. Calculate OM histograms by soft quantization
			3. Decide cluster number Kg by choosing most frequently occurring OM features by ensuring they cover 95% of histogram distributions of all pixels in the input images 
			4. calculate cluster saliency with spatial priors
*/
void calcSaliencyByOM( Mat& imgSal, Mat& idx, Mat& rank, Mat& variance, const Mat& img8u, const Mat& img32f, const IParams& params )
{
	Mat mag, grad, orient, resizedImg8u ;
	Mat OMResponse ;

	getGradFeatures( mag, grad, orient, resizedImg8u, img8u, Size( params.winLen, params.winLen ) ) ;
	getOMResponse( OMResponse, orient, grad, mag, params.winLen ) ;

	int strucCenterNum = getClusterNumByQuantize4Struct( orient, mag, OrientationBinNum, 0.95f ) / params.omClusterFactor ;

	calcClusterSaliencyWithSpatialPriors( imgSal, idx, rank, variance, OMResponse, strucCenterNum, img32f, params ) ;
}
