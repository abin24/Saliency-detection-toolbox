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

#include "superpixel.h"
#include <limits>
#include <cstdio>
#include <random>
#include <functional>

SuperpixelStatistic::SuperpixelStatistic() : mean_color_( 0.f, 0.f, 0.f ), mean_position_( 0.f, 0.f ) {
}

Superpixel::Superpixel( int K, float col_w, int n_iter, bool geodesic ) : K_( K ), col_w_( col_w ), n_iter_(n_iter), geodesic_(geodesic) {
}

Mat_< int > Superpixel::segment( const Mat_< Vec3f >& im ) const {
	if (geodesic_)
		return geodesicSegmentation( im );
	else
		return slic( im );
}
Mat_< int > Superpixel::geodesicSegmentation( const Mat_< Vec3f >& im ) const {
	std::uniform_int_distribution<int> distribution(-2, 2);
	std::mt19937 engine; // Mersenne twister MT19937
	auto randint = std::bind(distribution, engine);
	
	// Compute the spacing and grid size of the superpixels
	double sp_area = 1.0 * im.cols * im.rows / K_;
	int Kx = 0.5 + im.cols / sqrt( sp_area ), Ky = 0.5 + im.rows / sqrt( sp_area );
	int K = Kx*Ky;
	
	int win_sz = 1.0 * sqrt(sp_area) + 1;
	
	// Initialize the seeds on a regular grid
	std::vector< int64 > cnt( K );
	std::vector< Point2d > seedsd( K );
	std::vector< Point > seeds( K );
	for( int i=0,k=0; i<Kx; i++ )
		for( int j=0; j<Ky; j++, k++ )
			seeds[k] = Point( (i+0.5)*(im.cols-1)/Kx, (j+0.5)*(im.rows-1)/Ky ) + Point( randint(), randint() );
	
	Mat_<float> dx( im.size() ), dy( im.size() );
	
	for( int j=0; j<im.rows; j++ )
		for( int i=0; i<im.cols; i++ ) {
			if (i)
				dx(j,i-1) = col_w_*sqrt( (im(j,i)-im(j,i-1)).dot(im(j,i)-im(j,i-1)) ) + 1;
			if (j)
				dy(j-1,i) = col_w_*sqrt( (im(j-1,i)-im(j,i)).dot(im(j-1,i)-im(j,i)) ) + 1;
		}
	
	// Run k-means
	Mat_<float> dist( im.size() );
	Mat_<int> label( im.size() );
	for( int it=0; it<n_iter_; it++ ) {
		// Assignment step
		dist = std::numeric_limits<float>::max();
		label = -1;
		for( int k=0; k<K; k++ ) {
			dist( seeds[k] ) = 0;
			label( seeds[k] ) = k;
		}
		for( int IT=0; IT<2; IT++ ){
			for( int j=0; j<im.rows; j++ )
				for( int i=0; i<im.cols; i++ ) {
					if (i && dist(j,i-1) + dx(j,i-1) < dist(j,i)) {
						dist(j,i) = dist(j,i-1) + dx(j,i-1);
						label(j,i) = label(j,i-1);
					}
					if (j && dist(j-1,i) + dx(j-1,i) < dist(j,i)) {
						dist(j,i) = dist(j-1,i) + dy(j-1,i);
						label(j,i) = label(j-1,i);
					}
				}
			for( int j=im.rows-1; j>=0; j-- )
				for( int i=im.cols-1; i>=0; i-- ) {
					if (i && dist(j,i) + dx(j,i-1) < dist(j,i-1)) {
						dist(j,i-1) = dist(j,i) + dx(j,i-1);
						label(j,i-1) = label(j,i);
					}
					if (j && dist(j,i) + dx(j-1,i) < dist(j-1,i)) {
						dist(j-1,i) = dist(j,i) + dy(j-1,i);
						label(j-1,i) = label(j,i);
					}
				}
		}
		for( int k=0; k<K; k++ ) {
			Vec3f c = im( seeds[k] );
			for( int j=std::max(0,seeds[k].y-win_sz); j<im.rows && j<=seeds[k].y+win_sz; j++ )
				for( int i=std::max(0,seeds[k].x-win_sz); i<im.cols && i<=seeds[k].x+win_sz; i++ ){
					double d = (i-seeds[k].x) * (i-seeds[k].x) + (j-seeds[k].y) * (j-seeds[k].y);
					double cd = ( im( j, i ) - c ).dot( im( j, i ) - c );
					d += col_w_ * col_w_ * cd;
					if( d < dist( j, i ) ) {
						dist( j, i ) = d;
						label( j, i ) = k;
					}
				}
		}
		
		// Update
		for( int k=0; k<K; k++ ) {
			seedsd[k] = Point2d(0,0);
			cnt[k] = 0;
		}
		for( int j=0; j<im.rows; j++ )
			for( int i=0; i<im.cols; i++ ) {
				// Fix all the pixels we messed up!
				if ( label( j, i ) < 0 ) {
// 					printf("Oops that wasn't very slick :(\n");
					for( int k=0; k<K; k++ ){
						Vec3f c = im( seeds[k] );
						double d = (i-seeds[k].x) * (i-seeds[k].x) + (j-seeds[k].y) * (j-seeds[k].y);
						double cd = ( im( j, i ) - c ).dot( im( j, i ) - c );
						d += col_w_ * col_w_ * cd;
						if( d < dist( j, i ) ) {
							dist( j, i ) = d;
							label( j, i ) = k;
						}
					}
				}
				
				seedsd[ label( j, i ) ] += Point2d( i, j );
				cnt[ label( j, i ) ] += 1;
			}
		
		for( int k=0; k<K; k++ )
			if (cnt[k] > 0)
				seeds[k] = Point( 0.5 + seedsd[k].x / cnt[k], 0.5 + seedsd[k].y / cnt[k] );
	}
	return label;
}
// We are implementing SLIC here. I'm too lazy to enfoce the connectivity
Mat_< int > Superpixel::slic( const Mat_< Vec3f >& im ) const {
	// Compute the spacing and grid size of the superpixels
	double sp_area = 1.0 * im.cols * im.rows / K_;
	int Kx = 0.5 + im.cols / sqrt( sp_area ), Ky = 0.5 + im.rows / sqrt( sp_area );
	int K = Kx*Ky;
	
	int win_sz = 1.0 * sqrt(sp_area) + 1;
	
	// Initialize the seeds on a regular grid
	std::vector< int64 > cnt( K );
	std::vector< Point2d > seedsd( K );
	std::vector< Point > seeds( K );
	for( int i=0,k=0; i<Kx; i++ )
		for( int j=0; j<Ky; j++, k++ )
			seeds[k] = Point( (i+0.5)*(im.cols-1)/Kx, (j+0.5)*(im.rows-1)/Ky );
	
	// Run k-means
	Mat_<float> dist( im.size() );
	Mat_<int> label( im.size() );
	for( int it=0; it<n_iter_; it++ ) {
		// Assignment step
		dist = std::numeric_limits<float>::max();
		label = -1;
		for( int k=0; k<K; k++ ) {
			Vec3f c = im( seeds[k] );
			for( int j=std::max(0,seeds[k].y-win_sz); j<im.rows && j<=seeds[k].y+win_sz; j++ )
				for( int i=std::max(0,seeds[k].x-win_sz); i<im.cols && i<=seeds[k].x+win_sz; i++ ){
					double d = (i-seeds[k].x) * (i-seeds[k].x) + (j-seeds[k].y) * (j-seeds[k].y);
					double cd = ( im( j, i ) - c ).dot( im( j, i ) - c );
					d += col_w_ * col_w_ * cd;
					if( d < dist( j, i ) ) {
						dist( j, i ) = d;
						label( j, i ) = k;
					}
				}
		}
		
		// Update
		for( int k=0; k<K; k++ ) {
			seedsd[k] = Point2d(0,0);
			cnt[k] = 0;
		}
		for( int j=0; j<im.rows; j++ )
			for( int i=0; i<im.cols; i++ ) {
				// Fix all the pixels we messed up!
				if ( label( j, i ) < 0 ) {
// 					printf("Oops that wasn't very slick :(\n");
					for( int k=0; k<K; k++ ){
						Vec3f c = im( seeds[k] );
						double d = (i-seeds[k].x) * (i-seeds[k].x) + (j-seeds[k].y) * (j-seeds[k].y);
						double cd = ( im( j, i ) - c ).dot( im( j, i ) - c );
						d += col_w_ * col_w_ * cd;
						if( d < dist( j, i ) ) {
							dist( j, i ) = d;
							label( j, i ) = k;
						}
					}
				}
				
				seedsd[ label( j, i ) ] += Point2d( i, j );
				cnt[ label( j, i ) ] += 1;
			}
		
		for( int k=0; k<K; k++ )
			if (cnt[k] > 0)
				seeds[k] = Point( 0.5 + seedsd[k].x / cnt[k], 0.5 + seedsd[k].y / cnt[k] );
	}
	return label;
}

std::vector< SuperpixelStatistic > Superpixel::stat( const Mat_< Vec3f >& im, const Mat_< Vec3b >& rgb, const Mat_< int >& segmentation ) const {
	int K = nLabels( segmentation );
	std::vector< SuperpixelStatistic > stat( K );
	std::vector< double > cnt( K, 1e-10 );
	
	for( int j=0; j<im.rows; j++ )
		for( int i=0; i<im.cols; i++ ) {
			int l = segmentation( j, i );
			if ( l >=0 ) {
				stat[ l ].mean_color_ += im(j,i);
				stat[ l ].mean_rgb_ += rgb(j,i);
				stat[ l ].mean_position_ += Vec2f( i, j );
				cnt[ l ] += 1;
			}
		}
	for( int i=0; i<K; i++ ) {
		stat[ i ].mean_color_ *= 1.0 / cnt[ i ];
		stat[ i ].mean_rgb_ *= 1.0 / cnt[ i ];
		stat[ i ].mean_position_ *= 1.0 / cnt[ i ];
		stat[ i ].size_ = cnt[ i ];
	}
	// Rescale the position parameter
	for( int i=0; i<K; i++ )
		stat[ i ].mean_position_ *= 1.0 / std::max( im.cols, im.rows );
	return stat;
}

Mat_< Vec3f > Superpixel::visualizeMeanColor( const Mat_< int >& segmentation, const std::vector< SuperpixelStatistic >& stat ) const {
	std::vector< Vec3f > colors;
	for( int k=0; k<stat.size(); k++ )
		colors.push_back( stat[k].mean_color_ );
	return assign( colors, segmentation );
}
Mat_< Vec3f > Superpixel::visualizeRandom( const Mat_< int >& segmentation ) const {
	srand(segmentation.size().area());
	
	int n_label = nLabels( segmentation );
	
	std::vector< Vec3f > colors;
	for( int k=0; k<n_label; k++ )
		colors.push_back(Vec3f(1.0*rand() / RAND_MAX, 1.0*rand() / RAND_MAX, 1.0*rand() / RAND_MAX));
	
	return assign( colors, segmentation );
}
int Superpixel::nLabels( const cv::Mat_< int >& segmentation ) const {
	int n_label = 0;
	for( int j=0; j<segmentation.rows; j++ )
		for( int i=0; i<segmentation.cols; i++ )
			if ( n_label <= segmentation(j,i) )
				n_label = segmentation(j,i) + 1;
	return n_label;
}
