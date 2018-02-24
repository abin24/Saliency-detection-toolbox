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

#include "SF.h"
#include "fastmath.h"
 
#include "./filter/filter.h"

SaliencySettings::SaliencySettings() {
	// Superpixel settings
	n_superpixels_ = 400;
	n_iterations_= 5;
	superpixel_color_weight_ = 1;
	
	// Saliency filter radii
	sigma_p_ = 0.25;
	sigma_c_ = 20.0;
	k_ = 3; // The paper states K=6 (but then I think we used standard deviation and not variance, was probably a typo)
	
	// Upsampling parameters
	min_saliency_ = 0.1;
	alpha_ = 1.0 / 30.0;
	beta_ = 1.0 / 30.0;
	
	// Various algorithm settings
	upsample_ = true;
	uniqueness_ = true;
	distribution_ = true;
	filter_uniqueness_ = filter_distribution_ = false;
	use_spix_color_ = false; // Disabled to get a slightly better performance
}


SF::SF( SaliencySettings settings ): settings_(settings), superpixel_( settings.n_superpixels_, settings.superpixel_color_weight_, settings.n_iterations_ ) {
}
Mat_< float > SF::saliency( const Mat_< Vec3b >& im ) const {
	// Convert the image to the lab space
	Mat_<Vec3f> rgbim, labim;
	im.convertTo( rgbim, CV_32F, 1.0/255. );
	cvtColor( rgbim, labim, CV_BGR2Lab );
	
	// Do the abstraction
	Mat_<int> segmentation = superpixel_.segment( labim );
	std::vector< SuperpixelStatistic > stat = superpixel_.stat( labim, im, segmentation );
	
	// Compute the uniqueness
	std::vector<float> unique( stat.size(), 1 );
	if (settings_.uniqueness_) {
		if (settings_.filter_uniqueness_)
			unique = uniquenessFilter( stat );
		else
			unique = uniqueness( stat );
	}
	
	// Compute the distribution
	std::vector<float> dist( stat.size(), 0 );
	if (settings_.distribution_) {
		if (settings_.filter_distribution_)
			dist = distributionFilter( stat );
		else
			dist = distribution( stat );
	}
	
	// Combine the two measures
	std::vector<float> sp_saliency( stat.size() );
	for( int i=0; i<stat.size(); i++ )
		sp_saliency[i] = unique[i] * exp( - settings_.k_ * dist[i] );
	
	// Upsampling
	Mat_<float> r;
	if (settings_.upsample_)
		r = assignFilter( im, segmentation, stat, sp_saliency );
	else
		r = assign( segmentation, sp_saliency );
	
	// Rescale the saliency to [0..1]
	double mn, mx;
	minMaxLoc( r, &mn, & mx );
	r = (r - mn) / (mx - mn);
	
	// Increase the saliency value until we are below the minimal threshold
	double m_sal = settings_.min_saliency_ * r.size().area();
	for( float sm = sum( r )[0]; sm < m_sal; sm = sum( r )[0] )
		r =  min( r*m_sal/sm, 1.0f );
	
	return r;
}

void SF::calculateSaliencyMap(Mat *src, Mat * dst)
{
	Mat_<Vec3f> rgbim, labim;
	(*src).convertTo(rgbim, CV_32F, 1.0 / 255.);
	cvtColor(rgbim, labim, CV_BGR2Lab);

	// Do the abstraction
	Mat_<int> segmentation = superpixel_.segment(labim);
	std::vector< SuperpixelStatistic > stat = superpixel_.stat(labim, *src, segmentation);

	// Compute the uniqueness
	std::vector<float> unique(stat.size(), 1);
	if (settings_.uniqueness_) {
		if (settings_.filter_uniqueness_)
			unique = uniquenessFilter(stat);
		else
			unique = uniqueness(stat);
	}

	// Compute the distribution
	std::vector<float> dist(stat.size(), 0);
	if (settings_.distribution_) {
		if (settings_.filter_distribution_)
			dist = distributionFilter(stat);
		else
			dist = distribution(stat);
	}

	// Combine the two measures
	std::vector<float> sp_saliency(stat.size());
	for (int i = 0; i<stat.size(); i++)
		sp_saliency[i] = unique[i] * exp(-settings_.k_ * dist[i]);

	// Upsampling
	Mat_<float> r;
	if (settings_.upsample_)
		r = assignFilter(*src, segmentation, stat, sp_saliency);
	else
		r = assign(segmentation, sp_saliency);

	// Rescale the saliency to [0..1]
	double mn, mx;
	minMaxLoc(r, &mn, &mx);
	r = (r - mn) / (mx - mn);

	// Increase the saliency value until we are below the minimal threshold
	double m_sal = settings_.min_saliency_ * r.size().area();
	for (float sm = sum(r)[0]; sm < m_sal; sm = sum(r)[0])
		r = min(r*m_sal / sm, 1.0f);

	*dst= r;

}

// Normalize a vector of floats to the range [0..1]
void normVec( std::vector< float > &r ){
	const int N = r.size();
	float mn = r[0], mx = r[0];
	for( int i=1; i<N; i++ ) {
		if (mn > r[i])
			mn = r[i];
		if (mx < r[i])
			mx = r[i];
	}
	for( int i=0; i<N; i++ )
		r[i] = (r[i] - mn) / (mx - mn);
}
std::vector< float > SF::uniqueness( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	std::vector< float > r( N );
	const float sp = 0.5 / (settings_.sigma_p_ * settings_.sigma_p_);
	for( int i=0; i<N; i++ ) {
		float u = 0, norm = 1e-10;
		Vec3f c = stat[i].mean_color_;
		Vec2f p = stat[i].mean_position_;
		
		// Evaluate the score, for now without filtering
		for( int j=0; j<N; j++ ) {
			Vec3f dc = stat[j].mean_color_ - c;
			Vec2f dp = stat[j].mean_position_ - p;
			
			float w = fast_exp( - sp * dp.dot(dp) );
			u += w*dc.dot(dc);
			norm += w;
		}
		// Let's not normalize here, must have been a typo in the paper
// 		r[i] = u / norm;
		r[i] = u;
	}
	normVec( r );
	return r;
}
std::vector< float > SF::distribution( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	std::vector< float > r( N );
	const float sc =  0.5 / (settings_.sigma_c_*settings_.sigma_c_);
	for( int i=0; i<N; i++ ) {
		float u = 0, norm = 1e-10;
		Vec3f c = stat[i].mean_color_;
		Vec2f p(0.f, 0.f);
		
		// Find the mean position
		for( int j=0; j<N; j++ ) {
			Vec3f dc = stat[j].mean_color_ - c;
			float w = fast_exp( - sc * dc.dot(dc) );
			p += w*stat[j].mean_position_;
			norm += w;
		}
		p *= 1.0 / norm;
		
		// Compute the variance
		for( int j=0; j<N; j++ ) {
			Vec3f dc = stat[j].mean_color_ - c;
			Vec2f dp = stat[j].mean_position_ - p;
			float w = fast_exp( - sc * dc.dot(dc) );
			u += w*dp.dot(dp);
		}
		r[i] = u / norm;
	}
	normVec( r );
	return r;
}
std::vector< float > SF::uniquenessFilter( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	
	// Setup the data and features
	std::vector< Vec2f > features( stat.size() );
	Mat_<float> data( stat.size(), 5 );
	for( int i=0; i<N; i++ ) {
		features[i] = stat[i].mean_position_ / settings_.sigma_p_;
		Vec3f c = stat[i].mean_color_;
		data(i,0) = 1;
		data(i,1) = c[0];
		data(i,2) = c[1];
		data(i,3) = c[2];
		data(i,4) = c.dot(c);
	}
	// Filter
	Filter filter( (const float*)features.data(), N, 2 );
	filter.filter( data.ptr<float>(), data.ptr<float>(), 5 );
	
	// Compute the uniqueness
	std::vector< float > r( N );
	for( int i=0; i<N; i++ ) {
		Vec3f c = stat[i].mean_color_;
		float u = 0, norm = 1e-10;
		Vec2f p = stat[i].mean_position_;
		
		r[i] = data(i,0)*c.dot(c) + data(i,4) - 2*( c[0]*data(i,1) + c[1]*data(i,2) + c[2]*data(i,3) );
	}
	normVec( r );
	return r;
}
std::vector< float > SF::distributionFilter( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	
	// Setup the data and features
	std::vector< Vec3f > features( stat.size() );
	Mat_<float> data( stat.size(), 4 );
	for( int i=0; i<N; i++ ) {
		features[i] = stat[i].mean_color_ / settings_.sigma_c_;
		Vec2f p = stat[i].mean_position_;
		data(i,0) = 1;
		data(i,1) = p[0];
		data(i,2) = p[1];
		data(i,3) = p.dot(p);
	}
	// Filter
	Filter filter( (const float*)features.data(), N, 3 );
	filter.filter( data.ptr<float>(), data.ptr<float>(), 4 );
	
	// Compute the uniqueness
	std::vector< float > r( N );
	for( int i=0; i<N; i++ )
		r[i] = data(i,3) / data(i,0) - ( data(i,1) * data(i,1) + data(i,2) * data(i,2) ) / ( data(i,0) * data(i,0) );
	
	normVec( r );
	return r;
}
Mat_< float > SF::assign( const Mat_< int >& seg, const std::vector< float >& sal ) const {
	return Superpixel::assign( sal, seg );
}
Mat_< float > SF::assignFilter( const Mat_< Vec3b >& im, const Mat_< int >& seg, const std::vector< SuperpixelStatistic >& stat, const std::vector< float >& sal ) const {
	std::vector< float > source_features( seg.size().area()*5 ), target_features( im.size().area()*5 );
	Mat_< Vec2f > data( seg.size() );
	// There is a type on the paper: alpha and beta are actually squared, or directly applied to the values
	const float a = settings_.alpha_, b = settings_.beta_;
	
	const int D = 5;
	// Create the source features
	for( int j=0,k=0; j<seg.rows; j++ )
		for( int i=0; i<seg.cols; i++, k++ ) {
			int id = seg(j,i);
			data(j,i) = Vec2f( sal[id], 1 );
			
			source_features[D*k+0] = a * i;
			source_features[D*k+1] = a * j;
			if (D == 5) {
				source_features[D*k+2] = b * stat[id].mean_rgb_[0];
				source_features[D*k+3] = b * stat[id].mean_rgb_[1];
				source_features[D*k+4] = b * stat[id].mean_rgb_[2];
			}
		}
	// Create the source features
	for( int j=0,k=0; j<im.rows; j++ )
		for( int i=0; i<im.cols; i++, k++ ) {
			target_features[D*k+0] = a * i;
			target_features[D*k+1] = a * j;
			if (D == 5) {
				target_features[D*k+2] = b * im(j,i)[0];
				target_features[D*k+3] = b * im(j,i)[1];
				target_features[D*k+4] = b * im(j,i)[2];
			}
		}
	
	// Do the filtering [Filtering using the target features twice works slightly better, as the method described in our paper]
	if (settings_.use_spix_color_) {
		Filter filter( source_features.data(), seg.cols*seg.rows, target_features.data(), im.cols*im.rows, D );
		filter.filter( data.ptr<float>(), data.ptr<float>(), 2 );
	}
	else {
		Filter filter( target_features.data(), im.cols*im.rows, D );
		filter.filter( data.ptr<float>(), data.ptr<float>(), 2 );
	}
	
	Mat_<float> r( im.size() );
	for( int j=0; j<im.rows; j++ )
		for( int i=0; i<im.cols; i++ )
			r(j,i) = data(j,i)[0] / (data(j,i)[1] + 1e-10);
	return r;
}






