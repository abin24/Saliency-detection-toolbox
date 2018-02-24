#include "AC.h"
template<typename T> inline T sqr(T x) { return x * x; }
//以上检测均是在R1 = 0, MinR2 = Min(Width, Height) / 8.MaxR2 = Min(Width, Height) / 2, Scale = 3的结果。
void AC::calculateSaliencyMap(Mat *src, Mat * dst)
{
	Mat img3f;
	(*src).convertTo(img3f, CV_32FC3, 1.0 / 255);
	Mat sal(img3f.size(), CV_32F), MeanR1, MeanR2;
	GaussianBlur(img3f, MeanR1, Size(3, 3), -1);
	cvtColor(MeanR1, MeanR1, CV_BGR2Lab);

	int Width = img3f.cols, Height = img3f.rows;
	int R1 = 0,Scale=3;
	if (R1 > 0)                                                                    //    如果R1=0，则表示就取原始像素
		blur(MeanR1, MeanR1, Size(R1, R1), Point(-1, -1));
	int  MinR2 = min(Width, Height) / 8, MaxR2 = min(Width, Height)/2  ;

	for (int Z = 0; Z < Scale; Z++)
	{
		MeanR1.copyTo(MeanR2);
		int radius = (MaxR2 - MinR2) * Z / (Scale - 1) + MinR2;
		if (radius % 2 == 0)
			radius++;
		blur(MeanR2, MeanR2, Size(radius, radius), Point(-1, -1));
		//MeanR2.convertTo();
		for (int r = 0; r < Height; r++)
		{
			float *s = sal.ptr<float>(r);
			float *lab = MeanR1.ptr<float>(r);
			float *lab2 = MeanR2.ptr<float>(r);
			for (int c = 0; c < Width; c++, lab += 3, lab2 += 3)
			{
				 
				s[c] += sqrt((float)(sqr(lab2[0] - lab[0]) + sqr(lab2[1] - lab[1]) + sqr(lab2[2] - lab[2])));
			}
				
		}
	}	 
	
	normalize(sal, *dst, 0, 1, NORM_MINMAX);
}