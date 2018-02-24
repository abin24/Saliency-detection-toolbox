/*
    Copyright (c) 2012, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

struct SuperpixelStatistic {
	Vec3f mean_color_;
	Vec3f mean_rgb_;
	Vec2f mean_position_;
	int size_;
	SuperpixelStatistic();
};

class Superpixel {
protected:
	int K_, n_iter_;
	float col_w_;
	bool geodesic_;
	Mat_<int> slic( const Mat_<Vec3f> & im ) const;
	Mat_<int> geodesicSegmentation( const Mat_<Vec3f> & im ) const;
public:
	Superpixel( int K, float col_w, int n_iter=5, bool geodesic=true );
	Mat_<int> segment( const Mat_<Vec3f> & im ) const;
	std::vector< SuperpixelStatistic > stat( const Mat_<Vec3f> & im, const Mat_< Vec3b >& rgb, const Mat_<int> & segmentation ) const;
	Mat_<Vec3f> visualizeMeanColor( const Mat_<int> & segmentation, const std::vector< SuperpixelStatistic > & stat ) const;
	Mat_<Vec3f> visualizeRandom( const Mat_<int> & segmentation ) const;
template< typename T >
	static Mat_<T> assign( const std::vector<T> & val, const Mat_<int> & segmentation );
	int nLabels( const Mat_<int> & segmentation ) const;
};

template< typename T >
Mat_<T> Superpixel::assign( const std::vector<T> & val, const Mat_<int> & segmentation ) {
	Mat_<T> r( segmentation.size() );
	for( int j=0; j<segmentation.rows; j++ )
		for( int i=0; i<segmentation.cols; i++ )
			r(j,i) = val[ segmentation(j,i) ];
	return r;
}
