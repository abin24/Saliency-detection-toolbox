#include "LC.h"
 
void LC::calculateSaliencyMap(Mat *src, Mat * dst)
{   
	
	Mat img;
	if ((*src).channels() == 3)
		cvtColor(*src, img, CV_BGR2GRAY);
	else
		img = *src;
	 
	double f[256], s[256];
	memset(f, 0, 256 * sizeof(double));
	memset(s, 0, 256 * sizeof(double));
	for (int r = 0; r < img.rows; r++)
	{
		uchar* data = img.ptr<uchar>(r);
		for (int c = 0; c < img.cols; c++)
			f[data[c]] += 1;//直方图f
	}
	double s_min = DBL_MAX, s_max = 0;

	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
			s[i] += abs(i - j) * abs(i - j) * f[j];//直方图的累积
		if (s[i]>s_max) s_max = s[i];
		if (s[i]<s_min) s_min = s[i];

	}
	//归一化到0-255
	for (int i = 0; i < 256; i++)
	{
		s[i] = (s[i] - s_min) / (s_max - s_min)*255;

	}


	Mat salimg(img.size(), CV_64F);
	for (int r = 0; r < img.rows; r++)
	{
		uchar* data = img.ptr<uchar>(r);
		double* sal = salimg.ptr<double>(r);
		for (int c = 0; c < img.cols; c++)
			sal[c] = s[data[c]];
	}
	//normalize(salimg, salimg, 0, 1, NORM_MINMAX, CV_32F);
	 
	salimg.convertTo(*dst, CV_8U, 1, 0);
}