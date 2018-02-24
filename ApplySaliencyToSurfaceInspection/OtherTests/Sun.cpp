#include "Sun.h"

void Sun::calculateSaliencyMap(Mat *src, Mat * dst)
{
	Mat imgf, tImg;
	imgf = *src;
	if (imgf.channels() == 3)
	{
		cvtColor(imgf, imgf, CV_RGB2GRAY);
	}
	imgf.convertTo(imgf, CV_32FC1, 1.0 / 255);
	Scalar colorM = mean(imgf);
	double gm = colorM[0];
	GaussianBlur(imgf, tImg, Size(3, 3), 0);

	Mat  Salft2(imgf.size(), CV_32F);
	for (int r = 0; r < tImg.rows; r++)
	{
		float *s = Salft2.ptr<float>(r);
		float *gray = tImg.ptr<float>(r);
		for (int c = 0; c < tImg.cols; c++)
		{
			if (gray[c]<gm)   //Assumptions 1 The defect is darker than normal parts
				s[c] = (  gray[c]-gm);
		}
			
	}
	exp(Salft2, Salft2);
	Salft2 = 1-Salft2;
	normalize(Salft2, Salft2, 0, 1, NORM_MINMAX);
	imshow("Salft2", Salft2);
	Mat imgblur;



	blur(imgf,imgblur,Size(17,17));
	Mat  SalLoacal(imgf.size(), CV_32F);
	for (int r = 0; r < tImg.rows; r++)
	{
		float *s = SalLoacal.ptr<float>(r);
		float *gray = tImg.ptr<float>(r);
		float*bl = imgblur.ptr<float>(r);
		for (int c = 0; c < tImg.cols; c++)
			if (gray[c]<gm)   //Assumptions 1 The defect is darker than normal parts 
			s[c] = (bl[c] - gray[c])*(bl[c] -gray[c]);
	}

	normalize(SalLoacal, SalLoacal, 0, 1, NORM_MINMAX);
	imshow("SalLoacal", SalLoacal);

	Mat  SalG(imgf.size(), CV_32F);
	for (int r = 0; r < tImg.rows; r++)
	{
		float *s = SalG.ptr<float>(r);
		float *gray = tImg.ptr<float>(r);
		for (int c = 0; c < tImg.cols; c++)
			if (gray[c]<gm)   //Assumptions 1 The defect is darker than normal parts 
			s[c] = (gm- gray[c])*(gm- gray[c]);
	}

	normalize(SalG, SalG, 0, 1, NORM_MINMAX);
	imshow("SalG", SalG);
 


	 




}