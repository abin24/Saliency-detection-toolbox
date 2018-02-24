 
/*This demo Reference to  Song, K. C., Hu, S. P., Yan, Y. H., & Li, J. (2014). 
Surface defect detection method using saliency linear scanning morphology for silicon 
steel strip under oil pollution interference. Isij International, 54(11), 2598-2607.

code by Yibin Huang
*/
//basice dependency
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <time.h>
#include<algorithm> 

//Classes  To include
#include "itti/itti.h"
#include "SR/SR.h"
#include "Rudinac/Rudinac.h"
#include"GMR/GMR.h"
#include "FT/FT.h"
#include "LC/LC.h"
#include "HC/HC.h"
#include "RC/RC.h"
#include "MBP/MBP.hpp"
#include "AC/AC.h"
#include"MSS/MSS.h"
#include"methodSun\Sun.h"
#include"methodSun/EntropyFil.h"
//extern Mat EntropyFiltFn(const Mat& image);


Mat img;
int th1 = 100, th2 = 30;
int th3= 15;
bool corlor = 0;//When this value is 1, it will be done on the corlor img as it is default
//in fact I find that in this case gray scale maybe better than LAB space
static void on_trackbar(int, void*)
{
 
   //step 1 compute saliecy map
	Mat gray;
	cvtColor(img,gray,CV_RGB2GRAY);
	
	Mat Sal, SalCanny;
	double Time = (double)cvGetTickCount();
	FT::calculateSaliencyMap(&gray, &Sal, corlor, 5);
	imshow("Fig5left", Sal);


	Mat Saluchar;
	Sal.convertTo(Saluchar, CV_8UC1, 255.0);
	Canny(Saluchar, SalCanny, th1, th2, 3);
	imshow("Fig6/a", SalCanny);
	int ksize = 2;  
	Mat Kernel = getStructuringElement(MORPH_CROSS,
		Size(2 * ksize + 1, 2 * ksize + 1),
		Point(ksize, ksize));    //MORPH_ELLIPSE


	//step 2 filt the img

	Mat Salfilted;
	morphologyEx(Saluchar, Salfilted, MORPH_OPEN, Kernel);
	morphologyEx(Salfilted, Salfilted, MORPH_CLOSE, Kernel);
	imshow("Fig6/b", Salfilted);
	Canny(Salfilted, SalCanny, th1, th2, 3);	
	imshow("Fig6/c", SalCanny);

	//step3 binarization
	Mat otus;
	threshold(Salfilted, otus, 0, 255, CV_THRESH_OTSU);
	imshow("Fig7/a otus", otus);
	Mat imbw = Salfilted > th3;
	imshow("Fig7/b", imbw);
	int maxValue = 255;
	int blockSize = 35;
	int constValue = 15;  
	cv::Mat local;
	cv::adaptiveThreshold(Salfilted, local, maxValue, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY , blockSize, constValue);	 
	imshow("local adaptiveThrshold", local);	
 
	//step 4 median blur;	
	medianBlur(imbw, imbw, 29);
	imshow("Fig7/c blur", imbw);
	//step 5 fill the defect
	//transpose(imbw, imbw );
	
	int step0 = imbw.step[0];
	int step1 = imbw.step[1];
	uchar *pbw = imbw.data;
	for (int k = 0; k < imbw.rows - 1; k += 1)
	{
		int lf = 10000, rt = 0;
		for (int j = 0; j < imbw.cols - 1; ++j)
		{
			if (pbw[k*step0 + j *step1]  )
			{
				if (j < lf)
					lf = j;
				else
					rt = j;

			}
		}
		for (int j = lf; j < rt; ++j)
		{
			pbw[k*step0 + j *step1] = 255;

		}
	}
	for (int k = 0; k < imbw.cols - 1; k += 1)
	{
		int lf = 10000, rt = 0;
		for (int j = 0; j < imbw.rows - 1; ++j)
		{
			if (pbw[j*step0 + k *step1]  )
			{
				if (j < lf)
					lf = j;
				else
					rt = j;

			}
		}
		for (int j = lf; j < rt; ++j)
		{
			pbw[j*step0 + k *step1] = 255;
			 
		}
	}
	imshow("Fig 9,b",imbw);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(imbw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	Time = (double)cvGetTickCount() - Time;
	Mat contoursImage(imbw.rows, imbw.cols, CV_8U, Scalar(0));
	for (int i = 0; i<contours.size(); i++){
		  drawContours(contoursImage, contours, i, Scalar(255), 30);
	}
	imshow("fig11/a", contoursImage);
 
	Mat region = gray.mul(imbw);
	medianBlur(region,region,3);
	imshow("fig9/e",region);
	
	Mat defect;
	Canny(region, defect, 232, 162, 3);
	imshow("fig9/f", defect);
	
	printf("run time = %f ms\n", Time / (cvGetTickFrequency() * 1000.0));// 
}
int main(int argc, char **argv)
{
	//read the src imgs
	
	img = imread("./Wipe_crack_1.bmp");
	imshow("img", img);
	createTrackbar("th1", "img", &th1, 255, on_trackbar);	 
	createTrackbar("th2", "img", &th2, 255, on_trackbar);
	createTrackbar("th3", "img", &th3, 255, on_trackbar);
	on_trackbar(th1, 0);  
	on_trackbar(th2, 0);	
	on_trackbar(th3, 0);
	while (char(waitKey(1)) != 'q') {}
	 

	return 0;
}
