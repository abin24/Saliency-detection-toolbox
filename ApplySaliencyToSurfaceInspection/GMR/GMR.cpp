/*************************************************

Copyright: Guangyu Zhong all rights reserved

Author: Guangyu Zhong

Date:2014-09-27

Description: codes for Manifold Ranking Saliency Detection
             Reference http://ice.dlut.edu.cn/lu/Project/CVPR13[yangchuan]/cvprsaliency.htm

**************************************************/
#include "GMR.h"
using namespace std;
typedef unsigned int UINT;

GMR::GMR()
{
	spNumMax = 200;
	compactness = 20.0;
	alpha = 0.99f;
	theta = 0.1f;
	spNum = 0;
}
GMR::~GMR()
{
}
Mat GMR::GeneSp(const Mat &img)
{
	int width = img.cols;
	int height = img.rows;
	int sz = width*height;
	UINT *reimg = new UINT[sz * 3];
	for (int c = 0; c<3; c++)
	{
		for (int i = 0; i<width; i++)
		{
			for (int j = 0; j<height; j++)

				reimg[c*(width*height) + i*height + j] = saturate_cast<unsigned int>(img.at<Vec3b>(j, i)[2 - c]);
		}
	}
	int* label = nullptr;
	SLIC slic;
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(reimg, height, width, label, spNum, spNumMax, compactness);
	Mat superpixels(img.size(), CV_16U);

	for (int i = 0; i<superpixels.rows; i++)
	{
		for (int j = 0; j<superpixels.cols; j++)
		{
			superpixels.at<ushort>(i, j) = label[i + j*superpixels.rows];
		}
	}
	delete [] reimg;
	delete [] label;
	return superpixels;
}

Mat GMR::GeneAdjMat(const Mat &spmat)
{
	int width = spmat.cols;
	int height = spmat.rows;
	Mat adjmat(Size(spmat.size()), CV_16U, Scalar(0));
	for (int i = 0; i < height - 1; ++i)
	{
		for (int j = 0; j < width - 1; ++j)
		{
			if (spmat.at<ushort>(i, j) != spmat.at<ushort>(i + 1, j))
			{
				adjmat.at<ushort>(spmat.at<ushort>(i, j), spmat.at<ushort>(i + 1, j)) = 1;
				adjmat.at<ushort>(spmat.at<ushort>(i + 1, j), spmat.at<ushort>(i, j)) = 1;
			}
			if (spmat.at<ushort>(i, j) != spmat.at<ushort>(i, j + 1))
			{
				adjmat.at<ushort>(spmat.at<ushort>(i, j), spmat.at<ushort>(i, j + 1)) = 1;
				adjmat.at<ushort>(spmat.at<ushort>(i, j + 1), spmat.at<ushort>(i, j)) = 1;
			}
			if (spmat.at<ushort>(i, j) != spmat.at<ushort>(i + 1, j + 1))
			{
				adjmat.at<ushort>(spmat.at<ushort>(i, j), spmat.at<ushort>(i + 1, j + 1)) = 1;
				adjmat.at<ushort>(spmat.at<ushort>(i + 1, j + 1), spmat.at<ushort>(i, j)) = 1;
			}
			if (spmat.at<ushort>(i + 1, j) != spmat.at<ushort>(i, j + 1))
			{
				adjmat.at<ushort>(spmat.at<ushort>(i + 1, j), spmat.at<ushort>(i, j + 1)) = 1;
				adjmat.at<ushort>(spmat.at<ushort>(i, j + 1), spmat.at<ushort>(i + 1, j)) = 1;
			}
		}
	}
	return adjmat;	
}
vector<int> GMR::GeneBdQuery(const Mat &superpixels, const int type)
{
	int height = superpixels.rows;
	int width = superpixels.cols;
	Mat transSuper = superpixels.t();
	vector<int> bd;
	switch (type)
	{
	case 1:
		bd = superpixels.row(0);
		break;
	case 2:
		bd = superpixels.row(height - 1);
		break;
	case 3:
		bd = transSuper.row(0);
		break;
	case 4:
		bd = transSuper.row(width - 1);
		break;
	case 5:
		vector<int> bdTop = superpixels.row(0);
		vector<int> bdDown = superpixels.row(height - 1);
		vector<int> bdRight = transSuper.row(0);
		vector<int> bdLeft = transSuper.row(width - 1);
		bd.insert(bd.end(), bdTop.begin(), bdTop.end());
		bd.insert(bd.end(), bdDown.begin(), bdDown.end());
		bd.insert(bd.end(), bdRight.begin(), bdRight.end());
		bd.insert(bd.end(), bdLeft.begin(), bdLeft.end());
		break;
	}
	sort(bd.begin(), bd.end());
	bd.erase(unique(bd.begin(), bd.end()), bd.end());
	return bd;
}

int GMR::GeneFeature(const Mat &img, const Mat &superpixels, const int feaType, Mat &feaSpL, Mat &feaSpA, Mat &feaSpB, Mat &spNpix, Mat &spCnt)
{
	Mat feaMap(img.size(),img.type());
	img.copyTo(feaMap);
	switch (feaType)
	{
	case 1:
		cvtColor(img, feaMap, CV_BGR2Lab);
		break;
	case 2:
		cvtColor(img, feaMap, CV_BGR2HSV);
		break;
	default:
		break;
	}
	/*vector<float> feaSpL(spNum, 0);
	vector<float> feaSpA(spNum, 0);
	vector<float> feaSpB(spNum, 0);
	vector<int> spNpix(spNum,0);*/
	//vector< vector<float> > spCnt(spNum, vector<float>(2, 0));  	
	for (int i = 0; i < superpixels.rows; ++i)
	{
		for (int j = 0; j < superpixels.cols; ++j)
		{
			feaSpL.at<float>(superpixels.at<ushort>(i, j)) += feaMap.at<Vec3b>(i, j)[0];
			feaSpA.at<float>(superpixels.at<ushort>(i, j)) += feaMap.at<Vec3b>(i, j)[1];
			feaSpB.at<float>(superpixels.at<ushort>(i, j)) += feaMap.at<Vec3b>(i, j)[2];
			spCnt.at<float>(superpixels.at<ushort>(i, j),0) += i;
			spCnt.at<float>(superpixels.at<ushort>(i, j),1) += j;
			++spNpix.at<float>(superpixels.at<ushort>(i, j),0);
		}
	}
	for (int i = 0; i < spNum; ++i)
	{
		feaSpL.at<float>(i) /= spNpix.at<float>(i);
		feaSpA.at<float>(i) /= spNpix.at<float>(i);
		feaSpB.at<float>(i) /= spNpix.at<float>(i);
		spCnt.at<float>(i, 0) /= spNpix.at<float>(i);
		spCnt.at<float>(i, 1) /= spNpix.at<float>(i);
	}
	/*double minv = 0;
	double maxv = 0;
	minMaxIdx(feaSpL, &minv, &maxv);
	feaSpL = (feaSpL - minv) / (maxv - minv);
	minMaxIdx(feaSpA, &minv, &maxv);
	feaSpA = (feaSpA - minv) / (maxv - minv);
	minMaxIdx(feaSpB, &minv, &maxv);
	feaSpB = (feaSpB - minv) / (maxv - minv);*/
	return 0;

}

Mat GMR::GeneWeight(const vector<float> &feaSpL, const vector<float> &feaSpA, const vector<float> &feaSpB, const Mat &superpixels, const vector<int> &bd, const Mat &adj)
{
	Mat weightMat(Size(spNum, spNum), CV_32F, Scalar(-1));
	int dist = 0;
	float minv = (float)numeric_limits<float>::max();
	float maxv = (float)numeric_limits<float>::min();
	for (int i = 0; i < spNum; ++i)
	{
		for (int j = 0; j < spNum; ++j)
		{
			if (adj.at<ushort>(i, j) == 1)
			{
				dist = sqrt(pow(feaSpL[i] - feaSpL[j], 2)) + sqrt(pow(feaSpA[i] - feaSpA[j], 2)) + sqrt(pow(feaSpB[i] - feaSpB[j], 2));
				weightMat.at<float>(i, j) = dist;
				if (dist < minv)
					minv = dist;
				if (dist > maxv)
					maxv = dist;

				for (int k = 0; k < spNum; ++k)
				{
					if (adj.at<ushort>(j, k) == 1)
					{
						dist = sqrt(pow(feaSpL[k] - feaSpL[i], 2)) + sqrt(pow(feaSpA[k] - feaSpA[i], 2)) + sqrt(pow(feaSpB[k] - feaSpB[i], 2));
						weightMat.at<float>(i, k) = dist;
						if (dist < minv)
							minv = dist;
						if (dist > maxv)
							maxv = dist;
					}

				}
			}

		}
	}

	for (int i = 0; i < bd.size(); ++i)
	{
		for (int j = 0; j < bd.size(); ++j)
		{
			dist = sqrt(pow(feaSpL[bd[i]] - feaSpL[bd[j]], 2)) + sqrt(pow(feaSpA[bd[i]] - feaSpA[bd[j]], 2)) + sqrt(pow(feaSpB[bd[i]] - feaSpB[bd[j]], 2));
			weightMat.at<float>(bd[i], bd[j]) = dist;
			if (dist < minv)
				minv = dist;
			if (dist > maxv)
				maxv = dist;
		}
	}

	for (int i = 0; i < spNum; ++i)
	{
		for (int j = 0; j < spNum; ++j)
		{
			if (weightMat.at<float>(i, j)>-1)
			{
				weightMat.at<float>(i, j) = (weightMat.at<float>(i, j) - minv) / (maxv - minv);
				weightMat.at<float>(i, j) = exp(-weightMat.at<float>(i, j) / theta);
			}
			else
				weightMat.at<float>(i, j) = 0;
		}
	}

	Mat tmpsuperpixels;
	normalize(weightMat, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(tmpsuperpixels, CV_8UC3, 1.0);
	//imshow("sp", tmpsuperpixels);
	//waitKey();
	return weightMat;
}

Mat GMR::GeneY(const vector<int> &bd)
{
	Mat Y(Size(spNum, 1), CV_32F, Scalar(0));
	for (int i = 0; i < bd.size(); ++i)
	{
		Y.at<float>(bd[i]) = 1;
	}
	return Y;
}

Mat GMR::inveMat(const Mat &weight, const Mat &Y)
{
	Mat D(Size(weight.size()), CV_32F, Scalar(0));
	Mat diagD(Size(weight.rows, 1), CV_32F);
	reduce(weight, diagD, 1, CV_REDUCE_SUM);
	D = Mat::diag(diagD);
	Mat LapMat(Size(weight.size()), CV_32F);
	LapMat = D - alpha*weight;
	Mat E = Mat::eye(weight.size(), CV_32F);
	Mat inverMat(Size(weight.size()), CV_32F);
	solve(LapMat, E, inverMat, CV_SVD);
	Mat tmpMat = Mat::ones(inverMat.size(), CV_32F) - Mat::eye(inverMat.size(), CV_32F);
	inverMat = inverMat.mul(tmpMat);
	Mat fData = inverMat*Y.t();
	return fData;
}

Mat GMR::Sal2Img(const Mat &superpixels, const Mat &Sal)
{
	Mat salMap(Size(superpixels.size()), CV_32F);
	for (int i = 0; i < superpixels.rows; ++i)
	{
		for (int j = 0; j < superpixels.cols; ++j)
		{
			salMap.at<float>(i, j) = Sal.at<float>(superpixels.at<ushort>(i, j));
		}
	}
	return salMap;
}
Mat GMR::GeneSal(const Mat &img)
{
	Mat superpixels = GeneSp(img);
	Mat feaSpL(Size(1, spNum), CV_32F, Scalar(0));
	Mat feaSpA(Size(1, spNum), CV_32F, Scalar(0));
	Mat feaSpB(Size(1, spNum), CV_32F, Scalar(0));
	Mat spNpix(Size(1, spNum), CV_32F, Scalar(0));
	Mat spCnt(Size(2, spNum), CV_32F, Scalar(0));
	GeneFeature(img, superpixels, 1, feaSpL, feaSpA, feaSpB, spNpix, spCnt);
	Mat adjmat = GeneAdjMat(superpixels);
	vector<int> bd = GeneBdQuery(superpixels, 5);
	Mat affmat =  GeneWeight(feaSpL, feaSpA, feaSpB, superpixels, bd, adjmat);
	Mat Y = GeneY(bd);
	Mat Sal = inveMat(affmat, Y);
	double minv = 0;
	double maxv = 0;
	minMaxIdx(Sal, &minv, &maxv);
	Sal = (Sal - minv) / (maxv - minv);
	Sal = 1 - Sal;
	return Sal;
}
void GMR::calculateSaliencyMap(Mat *img, Mat* dst)
{
	Mat superpixels =  GeneSp(*img);
	Mat sal = GeneSal(*img);

	Mat salMap = Sal2Img(superpixels, sal);
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(*dst, CV_8UC1, 1.0);
}
