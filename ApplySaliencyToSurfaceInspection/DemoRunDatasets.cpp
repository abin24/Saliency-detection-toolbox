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
#include"BMS/BMS.h"
#include "SF/SF.h"
typedef vector<string> vecS;
#define charPointers2StrVec(arrayOfCharPointer) (vecS(arrayOfCharPointer, std::end(arrayOfCharPointer)))



int corlor = 0;//if for the gray image,itti cannot deal with its corlor feature map


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

int main(int argc, char **argv)
{
	ITTI itti;	SR sr; Rudinac rudi; GMR gmr; MSS mss; AC ac; MBP mbp; RC rc; HC hc; LC lc; FT ft; SF sf;
	//"AC", "FT", "GMR", "HC", "LC", "MBP", "MSS", "RC", "Rudinac", "SR", "ITTI","PHOT"
	const char* _methodNames[] = { "SF", "BMS", "SR"
		//"AC", "FT", "GMR", "HC", "LC", "MBP", "MSS", "RC", "Rudinac", "SR", "ITTI", "PHOT", "BMS", "SR"
	};//  
	const char* _dbNames[] = { "DAGM_Class1","DAGM_Class2"};
	//"RoadCracks""NEU_Oil", "NEU_SDI", "NEU_SPDI", "MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free", "DAGM_Class1"
	//, "DAGM_Class2", "DAGM_Class3", "DAGM_Class4", "DAGM_Class5", "DAGM_Class6", "DAGM_Class7", "DAGM_Class8", "DAGM_Class9", "DAGM_Class10"

	vecS dbNames = charPointers2StrVec(_dbNames);
	vecS methodNames = charPointers2StrVec(_methodNames);
	string rootDir = "E:/datasets/SaliecyDataset/Surface/";




	for (int i = 0; i < dbNames.size(); i++)
	{       //loop the databases
		string wkDir = rootDir + dbNames[i] + "/";
		string salDir = wkDir + "/Saliency/";
		std::printf("Processing dataset: %s\n", (dbNames[i]));
		corlor = 1;
		string st = dbNames[i];
		corlor = st.find("MT");
		if (corlor != string::npos||corlor<0)//if it is MT it is gray value image。
		corlor = 0;	

		corlor = st.find("DAGM");		 
		if (corlor != string::npos || corlor<0)//if it is DAGM it is gray value image。
	    corlor = 0;

		vector<string> files;
		vector<double> TimeUse;
		int imnum = GetNames(wkDir + "Imgs/", "jpg", files);
		//glob(imgNameW, files, false); //为false时，仅仅遍历指定文件夹内符合模式的文件，当recursive为true时，会同时遍历指定文件夹的子文件夹。 opencv自带
		for (int k = 0; k < methodNames.size(); k++)
		{
			double Time = (double)cvGetTickCount();
			for (int j = 0; j < imnum; j++)//for (int j = 0; j < imnum; j++)
			{

				string ImgNane = wkDir + "Imgs/" + files[j];
				Mat img = imread(ImgNane);
				string salname = salDir + GetNameNE(files[j]);
				/*imshow("img", img);
				waitKey(1);*/
				Mat dst = cv::Mat::zeros(Size(img.cols, img.rows), CV_8UC1);//The size should be the same

				if (methodNames.at(k).compare("PHOT") == 0)
				{
					PHOT::calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_PHOT.png", dst * 255);
				}
				if (methodNames.at(k).compare("ITTI") == 0)
				{
					itti.calculateSaliencyMap(&img, &dst, corlor);
					imwrite(salname + "_ITTI.png", dst * 255);
				}

				if (methodNames.at(k).compare("SR") == 0)
				{
					sr.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_SR.png", dst * 255);
				}
				if (methodNames.at(k).compare("GMR") == 0)
				{
					gmr.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_GMR.png", dst );
				}


				if (methodNames.at(k).compare("FT") == 0)
				{
					ft.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_FT.png", dst * 255);

				}

				if (methodNames.at(k).compare("LC") == 0)
				{
					lc.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_LC.png", dst );
				}

				if (methodNames.at(k).compare("HC") == 0)
				{
					hc.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_HC.png", dst * 255);
				}

				if (methodNames.at(k).compare("RC") == 0)
				{
					rc.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_RC.png", dst * 255);
				}
				if (methodNames.at(k).compare("MBP") == 0)
				{
					mbp.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_MBP.png", dst );
				}
				if (methodNames.at(k).compare("AC") == 0)
				{
					ac.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_AC.png", dst * 255);
				}
				if (methodNames.at(k).compare("MSS") == 0)
				{
					mss.calculateSaliencyMap(&img, &dst);
					imwrite(salname + "_MSS.png", dst * 255);
				}
				if (methodNames.at(k).compare("Rudinac") == 0)
				{
					rudi.calculateSaliencyMap(&img, &dst,  corlor);
					 
					imwrite(salname + "_Rudinac.png", dst * 255);
				}
				if (methodNames.at(k).compare("BMS") == 0)
				{ 
					BMS::calculateSaliencyMap(&img, &dst);
					/*imshow("img", img);
  					imshow("BMS", dst);
					waitKey(); */
					imwrite(salname + "_BMS.png", dst  );
				}
				if (methodNames.at(k).compare("SF") == 0)
				{
					sf.calculateSaliencyMap(&img, &dst);
				 
					/*imshow("SF", dst);
					waitKey();*/
					imwrite(salname + "_SF.png", dst*255  );
				}


			}
			Time = (double)cvGetTickCount() - Time;
			TimeUse.push_back(Time / cvGetTickFrequency());
			std::cout << methodNames.at(k);
			std::printf("method total time = %f s\n", Time / (cvGetTickFrequency()));
			std::printf("each image used time = %f ms\n", Time / (cvGetTickFrequency() * 1000.0) / imnum);

		}//for k 

	}
	 
	waitKey();


	return 0;
}
