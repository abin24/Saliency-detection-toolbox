#include <io.h>
#include<windows.h>
//basic  dependency
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
#include"PHOT\PHOT.h"
#include "BMS/BMS.h"

typedef vector<string> vecS;
#define charPointers2StrVec(arrayOfCharPointer) (vecS(arrayOfCharPointer, std::end(arrayOfCharPointer)))
bool corlor = 0;//if for the gray image,itti cannot deal with its corlor feature map
int GetNames(string path, string exd, vector<string>& names)
{
	names.clear();
	names.reserve(10000);
	string nameW;

	if (0 != strcmp(exd.c_str(), ""))
	{
		nameW = path + "/*." + exd;
	}

	else
	{
		nameW = path + "/*";
	}


	WIN32_FIND_DATAA fileFindData;
	HANDLE hFind = ::FindFirstFileA(nameW.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do{
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue; // Ignore sub-folders
		names.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)names.size();
}
string GetNameNE(string& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start, path.find_last_not_of(' ') + 1 - start);
}

Mat SmallSuppression(Mat src,double th=0.5, double k=0.5)
{
 
    Mat  dst(Size(src.cols, src.rows), CV_32F);
	  
	for (int r = 0; r < src.rows; r++)
	{
		float *s = src.ptr<float>(r);
		float *d = dst.ptr<float>(r);
		for (int c = 0; c < src.cols; c++ )
			if (s[c]>th)
			d[c] = s[c];
			else
			d[c] = s[c]*k;

	}
	return dst;

}

int main(int argc, char **argv)
{
	ITTI itti;	SR sr; Rudinac rudi; GMR gmr; MSS mss; AC ac; MBP mbp; RC rc; HC hc; LC lc; FT ft;  
	//"AC", "FT", "GMR", "HC", "LC", "MBP", "MSS", "RC", "Rudinac", "SR", "ITTI"
	//"MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free", "DAGM_Class1" "MT_Crack","MT_Blowhole"
	const char* _dbNames[] = { "MT_Fray", "MT_CrackAndBlowhole"
		  
	};
	//"RoadCracks" ,"NEU_Oil","NEU_SDI","NEU_SPDI", "MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free"	 ，"DAGM_Class1"
	//, "DAGM_Class2", "DAGM_Class3", "DAGM_Class4", "DAGM_Class5", "DAGM_Class6", "DAGM_Class7", "DAGM_Class8", "DAGM_Class9", "DAGM_Class10"
	
	
	//Clue 1 The defects are always darker 
	//Clue 2 The defects are always with gradien changes
	//Clue 3 The defects myabe diffrent from the defects free palces and its neighbor

	//double sigma = -0.50;

	vecS dbNames = charPointers2StrVec(_dbNames);
	string rootDir = "E:/datasets/SaliecyDataset/Surface/";



	for (int i = 0; i < dbNames.size(); i++)
	{   
		
		//循环每个dateset  root each dateset
		string wkDir = rootDir + dbNames[i] + "/";
		string salDir = wkDir + "/Saliency/";
		std::printf("Processing dataset: %s\n", (dbNames[i]));
		vector<string> files;
		vector<double> TimeUse;
		int imnum = GetNames(wkDir + "Imgs/", "jpg", files);
	 	
		for (int j = 0; j < imnum; j++)//for (int j = 0; j < imnum; j++)
		{
			double Time = (double)cvGetTickCount();
			string ImgNane = wkDir + "Imgs/" + files[j];
			string salname = salDir + GetNameNE(files[j]);


			//////////////////////////////////////start Mcue//////////////////////////////////////////
			Mat img = imread(ImgNane);
			Mat gray;
			cvtColor(img, gray, CV_RGB2GRAY);
			//imshow("img", img);
			Mat dst = cv::Mat::zeros(Size(img.cols, img.rows), CV_32F);//The size should be the same			
			//Mat MCueSalSupress = cv::Mat::zeros(Size(img.cols, img.rows), CV_32F);//The size should be the same
			//harrisslike saliency		

			int maxValue = 255;
			int blockSize = 35;
			int constValue = 15;  //局部阈值减去此值
			cv::Mat Darker;
			cv::adaptiveThreshold(gray, Darker, maxValue, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
			Darker.convertTo(Darker,CV_32F,1/255.0);
			 
		//get harrisLikeSaliecy
            int dsize = 4;
			imshow("Input",gray);
			resize(gray, gray, Size(gray.cols / dsize, gray.rows / dsize));
			Mat Ix, Iy;
			Sobel(gray, Ix, CV_32F, 1, 0, 3, 1, 1, BORDER_DEFAULT);
			Sobel(gray, Iy, CV_32F, 0, 1, 3, 1, 1, BORDER_DEFAULT);
			Mat Ix2, Iy2, Ixy;
			Ix2 = Ix.mul(Ix);
			Iy2 = Iy.mul(Iy);
			Ixy = Ix.mul(Iy);
			GaussianBlur(Ix2, Ix2, Size(5, 5), -1);
			GaussianBlur(Iy2, Iy2, Size(5, 5), 0);
			GaussianBlur(Ixy, Ixy, Size(5, 5), 0);
			Mat A, B;
			A = Ix2 - Iy2;
			A = A.mul(A) + 4 * Ixy.mul(Ixy);
			 
		//	pow(A,0.5,A); not nessarry
			B = Ix2 + Iy2;
			normalize(A , A, 0, 1, NORM_MINMAX, CV_32F, Mat());
			normalize(B , B, 0, 1, NORM_MINMAX, CV_32F, Mat());
			Mat Strukturtensor;
			addWeighted(A, 0.5, B, 0.5, 0, Strukturtensor);
			resize(Strukturtensor, Strukturtensor, Size(img.cols, img.rows));
			imshow("Contours", Strukturtensor);
			imshow("Darker", Darker);
			 

			
			 
			 
			PHOT::calculateSaliencyMap(&img, &dst);
			imshow("PHOT", dst);
			Mat dstPHOT = dst.clone();			
			 
			ac.calculateSaliencyMap(&img, &dst);
			Mat dstAC = dst.clone();
			imshow("ac", dst);
			

			BMS::calculateSaliencyMap(&img, &dst);
			Mat dstBMS = dst.clone();
			dstBMS.convertTo(dstBMS, CV_32F, 1 / 255.0);
			imshow("dstBMS", dstBMS);
			Mat MCueSal = cv::Mat::zeros(Size(img.cols, img.rows), CV_32F);//The size should be the same
			Darker = Darker * 3.0 + 1.0;
			MCueSal += dstPHOT * 3;
			MCueSal += dstAC;
			MCueSal += Strukturtensor;
			MCueSal += dstBMS*3;
			Mat M_Cue = MCueSal.mul(Darker) / 16;
			


			imwrite(salname + "_MCue.png", M_Cue * 255); 
			M_Cue.convertTo(M_Cue,CV_8UC1,255.0);
			imshow("M_Cue", M_Cue);


			
			//dstPHOT = dstPHOT * 3.5 + 0.5;
			Mat M_Cue2 = (dstBMS.mul(Darker.mul(dstPHOT * 3 + dstAC + Strukturtensor))) / 4;
			M_Cue2.convertTo(M_Cue2, CV_8UC1, 255.0);
			imshow("M_Cue2", M_Cue2);
			imwrite(salname + "_MCue2.png", M_Cue2 );
			//////////////////////////////////////end Mcue//////////////////////////////////////////
		 

			Time = (double)cvGetTickCount() - Time;
		//	TimeUse.push_back(Time / cvGetTickFrequency());

		 
			std::printf(" image used time = %f ms\n", Time / (cvGetTickFrequency() * 1000.0)  );
 			waitKey( 1  );

		}

	}

	waitKey();


	return 0;
}
