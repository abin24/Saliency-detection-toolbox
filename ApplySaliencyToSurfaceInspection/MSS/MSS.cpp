#include <stdlib.h>
#include <string.h>
#include <math.h>
 
#include "MSS.h"


using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MSS::MSS()
{

}

MSS::~MSS()
{

}
void MSS::calculateSaliencyMap(Mat *src, Mat * dst)
{
	int ChanNum = (*src).channels();
	if (ChanNum == 3)
	{
		Mat img3f;
		(*src).convertTo(img3f, CV_32FC3, 1.0 / 255);
		Mat sal(img3f.size(), CV_32F), tImg;		
		cvtColor(img3f, tImg, CV_BGR2Lab);
		GaussianBlur(tImg, tImg, Size(3, 3), 0);
		int height = img3f.rows;
		int width = img3f.cols;
		vector<Mat> lab;
		split(tImg, lab);
		Mat lvec = lab[0];
		Mat avec = lab[1];
		Mat bvec = lab[2];
		Mat lint, aint, bint;
		integral(lvec, lint,CV_32F); 
		integral(avec, aint, CV_32F);
		integral(bvec, bint, CV_32F);
		 
		for (int j = 0; j < height; j++)
		{
			int yoff = min(j, height - j);
			int y1 = max(j - yoff, 0);
			int y2 = min(j + yoff, height - 1);


			float*salmap = sal.ptr<float>(j);
			float*ls = lvec.ptr<float>(j);
			float*as = avec.ptr<float>(j);
			float*bs = bvec.ptr<float>(j);
			

			for (int k = 0; k < width; k++)
			{
				int xoff = min(k, width - k);
				int x1 = max(k - xoff, 0);
				int x2 = min(k + xoff, width - 1);
				double area = (x2 - x1 + 1) * (y2 - y1 + 1);
				double lval = (lint.at<float>(y2, x2) + lint.at<float>(y1, x1) - lint.at<float>(y2, x1) - lint.at<float>(y1, x2)) / area;
				double aval = (aint.at<float>(y2, x2) + aint.at<float>(y1, x1) - aint.at<float>(y2, x1) - aint.at<float>(y1, x2)) / area;
				double bval = (bint.at<float>(y2, x2) + bint.at<float>(y1, x1) - bint.at<float>(y2, x1) - bint.at<float>(y1, x2)) / area;

				salmap[k] =  (lval - ls[k]) * (lval - ls[k])
					+ (aval - as[k]) * (aval - as[k]) + (bval - bs[k])
					* (bval - bs[k]);//square of the euclidean distance
				 
					
				
				// cout << salmap[k] << endl;
			 
				 
			}
		}
		normalize(sal, *dst, 0, 1, NORM_MINMAX);
		 
	}
	
}