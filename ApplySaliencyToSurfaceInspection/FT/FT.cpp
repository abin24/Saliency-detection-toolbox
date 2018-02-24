#include "FT.h"
template<typename T> inline T sqr(T x) { return x * x; }
void FT::calculateSaliencyMap(Mat *src, Mat * dst, bool corlor,int ksize)
{
	if (corlor && (*src).channels() == 3)  //if  do it on corlor domain
	{
	
	Mat img3f = (*src);
	 
	img3f.convertTo(img3f, CV_32FC3, 1.0 / 255);
	Mat sal(img3f.size(), CV_32F), tImg;
	GaussianBlur(img3f, tImg, Size(ksize, ksize), 0);
	cvtColor(tImg, tImg, CV_BGR2Lab);
	Scalar colorM = mean(tImg);
	for (int r = 0; r < tImg.rows; r++)
	{
		float *s = sal.ptr<float>(r);
		float *lab = tImg.ptr<float>(r);
		for (int c = 0; c < tImg.cols; c++, lab += 3)
			s[c] = (float)(sqr(colorM[0] - lab[0]) + sqr(colorM[1] - lab[1]) + sqr(colorM[2] - lab[2]));
	}
	normalize(sal, *dst, 0, 1, NORM_MINMAX);
	}
	else
	{
		Mat imgf, tImg;
		imgf = *src;

		if (imgf.channels() == 3)
		{
			cvtColor(imgf, imgf, CV_RGB2GRAY);
		}
		imgf.convertTo(imgf, CV_32FC1, 1.0 / 255);
		Scalar colorM = mean(imgf);
		GaussianBlur(imgf, tImg, Size(ksize, ksize), 0);
		Mat  sal(imgf.size(), CV_32F);
		for (int r = 0; r < tImg.rows; r++)
		{
			float *s = sal.ptr<float>(r);
			float *gray = tImg.ptr<float>(r);
			for (int c = 0; c < tImg.cols; c++)
				s[c] = (colorM[0] - gray[c])*(colorM[0] - gray[c]);
		}

		normalize(sal, *dst, 0, 1, NORM_MINMAX);
	}
}