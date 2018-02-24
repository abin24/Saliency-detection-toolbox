#include <stdlib.h>
#include <string.h>
#include <math.h>
 
#include "PHOT.h"


using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

PHOT::PHOT()
{

}

PHOT::~PHOT()
{

}
void PHOT::calculateSaliencyMap(Mat *src, Mat * dst)
{
	int ChanNum = (*src).channels();
	Mat gray,   grayTemp, grayDown;
	if (ChanNum == 3)
	{
		cvtColor(*src, gray, CV_BGR2GRAY);
	}
	else
		gray = *src;
  
	vector<Mat> mv;
	

	Size imageSize(gray.cols, gray.rows);
	Mat realImage(imageSize, CV_64F);
	Mat imaginaryImage(imageSize, CV_64F); imaginaryImage.setTo(0);
	Mat combinedImage(imageSize, CV_64FC2);
	Mat imageDFT;
	Mat logAmplitude;
	Mat angle(imageSize, CV_64F);
	Mat magnitude(imageSize, CV_64F);
	Mat logAmplitude_blur;

	vector<Mat> Rp;
	Mat Real(imageSize, CV_64F);
	Mat phase(imageSize, CV_64F); imaginaryImage.setTo(0);
	Mat DFT, RI;
	gray.convertTo(Real, CV_64F);
	Rp.push_back(Real);
	Rp.push_back(phase);
	merge(Rp, RI);
	dft(RI, DFT);
	split(DFT, Rp);
	Mat MAG, PHASE;
	cartToPolar(Rp.at(0), Rp.at(1), MAG, PHASE, false);

	Rp.at(0) = Rp.at(0) / MAG;
	Rp.at(1) = Rp.at(1) / MAG;
	merge(Rp, RI);	 
	dft(RI, RI, CV_DXT_INVERSE_SCALE);//invert dft
	split(RI, Rp); 
	MAG = Rp.at(0);
	GaussianBlur(MAG, MAG, Size(7, 7), -1);
	Scalar MeanS = mean(MAG);
	double m = MeanS[1];
	for (int r = 0; r < MAG.rows; r++)
	{
		double *s = MAG.ptr<double>(r);
		for (int c = 0; c < MAG.cols; c++)
			s[c] = (s[c] - m)*(s[c] - m);
	}
	double minval, maxval;
	minMaxLoc(MAG, &minval, &maxval);
	//MAG = MAG / maxval;//normaliz	
	MAG.convertTo(*dst, CV_32F,1 / maxval);
		 
}