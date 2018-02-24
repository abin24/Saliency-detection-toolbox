#include "stdafx.h"

#pragma once ;

float normL2SqrFast(const float* v1, const float* v2, int n)
{
	__m128 a, b, c ;

	__m128 xfsSum = _mm_setzero_ps() ;
	for( int i = 0; i < n; i += 4 )
	{
		a = _mm_load_ps( v1 + i ) ;
		b = _mm_load_ps( v2 + i ) ;

		c = _mm_sub_ps( a, b ) ;
		c = _mm_mul_ps( c, c ) ;
		xfsSum = _mm_add_ps( xfsSum, c ) ;
	}

	const float* q = (const float*)&xfsSum ;
	return ( q[0] + q[1] + q[2] + q[3] ) ;
}


/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPPModified(const Mat& _data, Mat& _out_centers,
	int K, RNG& rng, int trials, const Mat& img32f, int* centers )
{
	int i, j, k, dims = _data.cols, N = _data.rows;
	const float* data = _data.ptr<float>(0);
	size_t step = _data.step/sizeof(data[0]);
	//vector<int> _centers(K);
	//int* centers = &_centers[0];

	vector<float> _dist(N*3);
	float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
	double sum0 = 0;

	centers[0] = (unsigned)rng % N;

	Vec3f* imgPixels = new Vec3f[N] ;

	for( i = 0; i < N; i++ )
	{
		imgPixels[i] = img32f.at<Vec3f>( i / img32f.cols, i % img32f.cols ) ;

		dist[i] = normL2SqrFast(data + step*i, data + step*centers[0], dims) * calcVec3fDist<float>( imgPixels[i], imgPixels[centers[0]] ) ;
		sum0 += dist[i];
	}

	for( k = 1; k < K; k++ )
	{
		double bestSum = DBL_MAX;
		int bestCenter = -1;

		for( j = 0; j < trials; j++ )
		{
			double p = (double)rng*sum0, s = 0;
			for( i = 0; i < N-1; i++ )
				if( (p -= dist[i]) <= 0 )
					break;
			int ci = i;
			Vec3f ciPixel = img32f.at<Vec3f>( ci / img32f.cols, ci % img32f.cols ) ;
			for( i = 0; i < N; i++ )
			{
				tdist2[i] = std::min( normL2SqrFast(data + step*i, data + step*ci, dims) 
					* calcVec3fDist<float>( imgPixels[i], ciPixel ), dist[i]);

				s += tdist2[i];
			}

			if( s < bestSum )
			{
				bestSum = s;
				bestCenter = ci;
				std::swap(tdist, tdist2);
			}
		}

		centers[k] = bestCenter;

		sum0 = bestSum;
		std::swap(dist, tdist);

	}
	delete [] imgPixels ;

	for( k = 0; k < K; k++ )
	{
		const float* src = data + step*centers[k];
		float* dst = _out_centers.ptr<float>(k);
		for( j = 0; j < dims; j++ )
			dst[j] = src[j];
	}
}

/*
	Random Seeding
*/
static void generateRandomCenter( const Mat& data, Mat& centers, int K, RNG& rng, const Mat& img32f, int* centerPos )
{
	int factor = 10 ;	
	int subRows = ( int )ceil( img32f.rows * 1.f / factor ) ;
	int subCols = ( int )ceil( img32f.cols * 1.f / factor ) ;
	cv::Mat subData = cv::Mat::zeros( subRows * subCols, data.cols, CV_32FC1 ) ;
	cv::Mat subImg32f = cv::Mat::zeros( subRows, subCols, CV_32FC3 ) ;

	int count = 0 ;
	for ( int row = 0; row < img32f.rows; row += factor )
	{
		for ( int col = 0; col < img32f.cols; col += factor )
		{
			memcpy( subData.ptr<float>( row / factor * subImg32f.cols + col / factor ), data.ptr<float>( row * img32f.cols + col ), sizeof( float ) * data.cols ) ;
			subImg32f.at<Vec3f>( row / factor, col / factor ) = img32f.at<Vec3f>( row, col ) ;
		}
	}

	generateCentersPPModified( subData, centers, K, rng, 3, subImg32f, centerPos ) ;
	for ( int k = 0; k < K; k++ )
	{
		int row = centerPos[k] / subImg32f.cols ;
		int col = centerPos[k] % subImg32f.cols ;
		centerPos[k] = row * factor * img32f.cols + col * factor ;
	}
}

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
double kmeansPPModified( InputArray _data, int K,
	InputOutputArray _bestLabels,
	TermCriteria criteria, int attempts,
	int flags, OutputArray _centers, const Mat& img )
{
	const int SPP_TRIALS = 3;
	Mat data = _data.getMat();
	bool isrow = data.rows == 1 && data.channels() > 1;
	int N = !isrow ? data.rows : data.cols;
	int dims = (!isrow ? data.cols : 1)*data.channels();
	int type = data.depth();

	Mat img32f ;
	if ( img.type() != CV_32FC3 )
		img.convertTo( img32f, CV_32FC3, 1 / 255.f ) ;
	else
		img32f = img ;
	//cvtColor( img32f, img32f, CV_RGB2Lab ) ;

	attempts = std::max(attempts, 1);
	CV_Assert( data.dims <= 2 && type == CV_32F && K > 0 );
	CV_Assert( N >= K );

	_bestLabels.create(N, 1, CV_32S, -1, true);

	Mat _labels, best_labels = _bestLabels.getMat();
	if( flags & CV_KMEANS_USE_INITIAL_LABELS )
	{
		CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
			best_labels.cols*best_labels.rows == N &&
			best_labels.type() == CV_32S &&
			best_labels.isContinuous());
		best_labels.copyTo(_labels);
	}
	else
	{
		if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
			best_labels.cols*best_labels.rows == N &&
			best_labels.type() == CV_32S &&
			best_labels.isContinuous()))
			best_labels.create(N, 1, CV_32S);
		_labels.create(best_labels.size(), best_labels.type());
	}
	int* labels = _labels.ptr<int>();

	int* centersPos = new int[K] ;
	Vec3f* centerColorVal = new Vec3f[K] ;

	Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
	vector<int> counters(K);
	vector<Vec2f> _box(dims);
	Vec2f* box = &_box[0];

	double best_compactness = DBL_MAX, compactness = 0;
	RNG& rng = theRNG();
	int a, iter, i, j, k;

	if( criteria.type & TermCriteria::EPS )
		criteria.epsilon = std::max(criteria.epsilon, 0.);
	else
		criteria.epsilon = FLT_EPSILON;
	criteria.epsilon *= criteria.epsilon;

	if( criteria.type & TermCriteria::COUNT )
		criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
	else
		criteria.maxCount = 100;

	if( K == 1 )
	{
		attempts = 1;
		criteria.maxCount = 2;
	}
	const float* sample = data.ptr<float>(0);


	for( j = 0; j < dims; j++ )
		box[j] = Vec2f(sample[j], sample[j]);

	for( i = 1; i < N; i++ )
	{
		sample = data.ptr<float>(i);
		for( j = 0; j < dims; j++ )
		{
			float v = sample[j];
			box[j][0] = std::min(box[j][0], v);
			box[j][1] = std::max(box[j][1], v);
		}
	}

	for( a = 0; a < attempts; a++ )
	{
		double max_center_shift = DBL_MAX;
		for( iter = 0;; )
		{
			swap(centers, old_centers);

			if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
			{
				if( flags & KMEANS_PP_CENTERS )
					generateCentersPPModified(data, centers, K, rng, SPP_TRIALS, img32f, centersPos );
				else
				{
					generateRandomCenter( data, centers, K, rng, img32f, centersPos ) ;
					//for( k = 0; k < K; k++ )
						//generateRandomCenter(_box, centers.ptr<float>(k), rng);
				}
			}			
			else
			{
				if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
				{
					for( i = 0; i < N; i++ )
						CV_Assert( (unsigned)labels[i] < (unsigned)K );
				}

				// compute centers
				centers = Scalar(0);
				for( k = 0; k < K; k++ )
					counters[k] = 0;

				for( i = 0; i < N; i++ )
				{
					sample = data.ptr<float>(i);
					k = labels[i];
					float* center = centers.ptr<float>(k);
					j=0;
#if CV_ENABLE_UNROLLED
					for(; j <= dims - 4; j += 4 )
					{
						float t0 = center[j] + sample[j];
						float t1 = center[j+1] + sample[j+1];

						center[j] = t0;
						center[j+1] = t1;

						t0 = center[j+2] + sample[j+2];
						t1 = center[j+3] + sample[j+3];

						center[j+2] = t0;
						center[j+3] = t1;
					}
#endif
					for( ; j < dims; j++ )
						center[j] += sample[j];
					counters[k]++;
				}

				if( iter > 0 )
					max_center_shift = 0;

				for( k = 0; k < K; k++ )
				{
					if( counters[k] != 0 )
						continue;

					// if some cluster appeared to be empty then:
					//   1. find the biggest cluster
					//   2. find the farthest from the center point in the biggest cluster
					//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
					int max_k = 0;
					for( int k1 = 1; k1 < K; k1++ )
					{
						if( counters[max_k] < counters[k1] )
							max_k = k1;
					}

					double max_dist = 0;
					int farthest_i = -1;
					float* new_center = centers.ptr<float>(k);
					float* old_center = centers.ptr<float>(max_k);
					float* _old_center = temp.ptr<float>(); // normalized
					float scale = 1.f/counters[max_k];
					for( j = 0; j < dims; j++ )
						_old_center[j] = old_center[j]*scale;

					for( i = 0; i < N; i++ )
					{
						if( labels[i] != max_k )
							continue;
						sample = data.ptr<float>(i);
						double dist = normL2Sqr_(sample, _old_center, dims) * calcVec3fDist<float>( img32f.at<Vec3f>( i / img32f.cols, i % img32f.cols ), img32f.at<Vec3f>( centersPos[max_k] / img32f.cols, centersPos[max_k] % img32f.cols ) )  ;

						if( max_dist <= dist )
						{
							max_dist = dist;
							farthest_i = i;
						}
					}

					counters[max_k]--;
					counters[k]++;
					labels[farthest_i] = k;
					sample = data.ptr<float>(farthest_i);

					for( j = 0; j < dims; j++ )
					{
						old_center[j] -= sample[j];
						new_center[j] += sample[j];
					}
				}

				for( k = 0; k < K; k++ )
				{
					float* center = centers.ptr<float>(k);
					CV_Assert( counters[k] != 0 );

					float scale = 1.f/counters[k];
					for( j = 0; j < dims; j++ )
						center[j] *= scale;

					if( iter > 0 )
					{
						double dist = 0;
						const float* old_center = old_centers.ptr<float>(k);
						for( j = 0; j < dims; j++ )
						{
							double t = center[j] - old_center[j];
							dist += t*t;
						}
						{
							max_center_shift = std::max(max_center_shift, dist);
						}

					}
				}
			}

			if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
				break;

			// assign labels
			for( int t = 0; t < K; t++ )
			{
				centerColorVal[t] = img32f.at<Vec3f>( centersPos[t] / img32f.cols, centersPos[t] % img32f.cols ) ;
			}

			compactness = 0;
			for( i = 0; i < N; i++ )
			{
				sample = data.ptr<float>(i);
				int k_best = 0;
				double min_dist = DBL_MAX;

				Vec3f pixelVal = img32f.at<Vec3f>( i / img32f.cols, i % img32f.cols ) ;

				for( k = 0; k < K; k++ )
				{
					const float* center = centers.ptr<float>(k);
					//double dist = normL2Sqr_(sample, center, dims) * calcVec3fDist<float>( pixelVal, centerColorVal[k] ) ;
					double dist = normL2SqrFast(sample, center, dims) * calcVec3fDist<float>( pixelVal, centerColorVal[k] ) ;

					if( min_dist > dist )
					{
						min_dist = dist;
						k_best = k;
					}
				}

				compactness += min_dist;
				labels[i] = k_best;
			}
		}
		if( compactness < best_compactness )
		{
			best_compactness = compactness;
			if( _centers.needed() )
				centers.copyTo(_centers);
			_labels.copyTo(best_labels);
		}
	}

	delete [] centersPos ;
	delete [] centerColorVal ;
	return best_compactness;
}

/*
	Regard histograms as vectors and use Kmeans++ to cluster them with adding color dissimilarity into distance measurement 
	Parameters:
		Input:
			hist: histograms to cluster
			kCenters: number of cluster centers for kmeans++ clustering
			img8u( CV_8UC3 ): original image for adding color dissimilarity into distance measurement of Kmeans++ 
		Output:
			idx: index map of histograms
			clusterHist: histograms of cluster center
			clusterWeight:  frequency of cluster center of the input histograms
*/
void clusterHistByKMeansPPModified( Mat& idx, Mat& clusterHist, Mat& clusterWeight, const Mat& hist, int kCenters, const Mat& img8u )
{
	kmeansPPModified( hist, kCenters, idx, TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 80, 1 ), 1, KMEANS_RANDOM_CENTERS, clusterHist, img8u ) ;

	clusterWeight = Mat::zeros( 1, clusterHist.rows, CV_32FC1 ) ;
	float* weight = (float*)clusterWeight.data ;
	int* dataPtr = idx.reshape( 0, 1 ).ptr<int>( 0 ) ;
	for ( int i = 0; i < idx.rows; i++ )
	{
		weight[dataPtr[i]]++ ;
	}
	normalize( clusterWeight, clusterWeight, 1, 0, NORM_L1, CV_32F ) ;
}


/*
	Regard histograms as vectors and use Kmeans++ to cluster them  
	Parameters:
		Input:
			hist: histograms to cluster
			kCenters: number of cluster centers for kmeans++ clustering
		Output:
			idx: index map of histograms
			clusterHist: histograms of cluster center
			clusterWeight:  frequency of cluster center of the input histograms
*/
void clusterHistByKMeansPP( Mat& idx, Mat& clusterHist, Mat& clusterWeight, const Mat& hist, int kCenters, const Mat& img8u )
{
	kmeans( hist, kCenters, idx, TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 80, 1 ), 1, KMEANS_PP_CENTERS, clusterHist ) ;

	clusterWeight = Mat::zeros( 1, clusterHist.rows, CV_32FC1 ) ;
	float* weight = (float*)clusterWeight.data ;
	int* dataPtr = idx.reshape( 0, 1 ).ptr<int>( 0 ) ;
	for ( int i = 0; i < idx.rows; i++ )
	{
		weight[dataPtr[i]]++ ;
	}
	normalize( clusterWeight, clusterWeight, 1, 0, NORM_L1, CV_32F ) ;
}