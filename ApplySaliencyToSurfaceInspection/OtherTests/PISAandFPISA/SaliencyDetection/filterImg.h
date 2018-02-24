#include "stdafx.h"
#include "CLMF_SDK_Import.h"

#pragma once

/*
	Saliency Coherence: smooth out _S and produce a spatially coherent yet discontinuity-preserving saliency map S.
	Parameters:
		Input:
			img8u: original image
			imgSal: image saliency result before filtered
		Output:
			dstSal: image saliency result after filtered
*/
void filterImg(  Mat& dstSal, const Mat& srcSal, const Mat& img8u, int filterOrder = 0, int radius = 15, float colorTau = 50, float epsl = 0.01f )
{
	FilterImageWithCLMF( 2, filterOrder, img8u, srcSal, dstSal, radius, colorTau, epsl, 1 );
	normalize( dstSal, dstSal, 0, 1, NORM_MINMAX ) ;
}

/*
	Propagate the saliency values among the pixels within the same cross support region to obtain a full-sized smoothly varying dense saliency map S for efficiency
	Parameters:
		Input:
			img8u: original color image as guidance
			imgSal: sparse saliency map Sl'
			points: pixels of the sparse image Il
			pWin: cross regions
		Output:
			dstSal: full-sized saliency map S
		Implement Details:
			Given a pixel p, its saliency value is obtained by its cross support region which contains p as well as the spatial distance from support pixels to pixel p.
*/
void jointlyUpsampleByCLMF( const Mat& img8u, Mat& dstSal, const Mat& srcSal, int maxArm, float colorTau )
{
	const int rows = srcSal.rows ;
	const int cols = srcSal.cols ;

	dstSal = Mat::zeros( img8u.size(), CV_32FC1 ) ;
	
	for( int rowS = 0, row = 1; rowS < rows; rowS++, row += stepSize )
	{
		const float* salPtr = srcSal.ptr<float>( rowS ) ;
		for( int colS = 0, col = 1; colS < cols; colS++, col += stepSize )
		{
			float val = salPtr[colS] ;
			for ( int y = row - 1; y <= row + 1; y++ )
			{
				float* dstSalPtr = dstSal.ptr<float>( y ) ;
				
				for ( int x = col - 1; x <= col + 1; x++ )
				{
					dstSalPtr[x] = val ;
				}
			} 
		}
	}	

	FilterImageWithCLMF( 2, 0, img8u, dstSal, dstSal, maxArm, colorTau, 0.01f, 1 );

	normalize( dstSal, dstSal, 0, 1, NORM_MINMAX ) ;
}