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
#include "./superpixel/superpixel.h"
#include<vector>

struct SaliencySettings{
	SaliencySettings();
	
	// Superpixel settings
	int n_superpixels_, n_iterations_;
	float superpixel_color_weight_;
	
	// Saliency filter radii
	float sigma_p_; // Radius for the uniqueness operator [eq 1]
	float sigma_c_; // Color radius for the distribution operator [eq 3]
	float k_; // The sharpness parameter of the exponential in the merging operation [eq 5]
	
	// Upsampling parameters
	float min_saliency_; // Minimum number of salient pixels for final rescaling
	float alpha_, beta_;
	
	// Various algorithm settings
	// Enable or disable parts of the algorithm
	bool upsample_, uniqueness_, distribution_, filter_uniqueness_, filter_distribution_;
	// Should we use the image color or superpixel color as a feature for upsampling
	bool use_spix_color_;
};

class SF {
protected:
	SaliencySettings settings_;
	Superpixel superpixel_;
protected:
	std::vector< float > uniqueness( const std::vector< SuperpixelStatistic > & stat ) const;
	std::vector< float > uniquenessFilter( const std::vector< SuperpixelStatistic > & stat ) const;
	std::vector< float > distribution( const std::vector< SuperpixelStatistic > & stat ) const;
	std::vector< float > distributionFilter( const std::vector< SuperpixelStatistic > & stat ) const;
	Mat_<float> assign( const Mat_<int> & seg, const std::vector< float > & sal ) const;
	
// 	std::vector< float > uniquenessFilter( const std::vector< SuperpixelStatistic > & stat ) const;
// 	std::vector< float > distributionFilter( const std::vector< SuperpixelStatistic > & stat ) const;
	Mat_<float> assignFilter( const Mat_<Vec3b> & im, const Mat_<int> & seg, const std::vector< SuperpixelStatistic > & stat, const std::vector< float > & sal ) const ;
public:
	SF( SaliencySettings settings = SaliencySettings() );
	Mat_<float> saliency(const Mat_<Vec3b> & im) const;
	void calculateSaliencyMap(Mat *src, Mat * dst);
};
