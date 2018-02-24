#include"RC.h"

template<typename T> inline T sqr(T x) { return x * x; }
typedef vector<string> vecS;
typedef vector<int> vecI;
typedef vector<float> vecF;
typedef vector<double> vecD;
typedef pair<double, int> CostIdx;
typedef pair<float, int> CostfIdx;
template<class T> inline T vecDist3(const Vec<T, 3> &v1, const Vec<T, 3> &v2) { return sqrt(sqr(v1[0] - v2[0]) + sqr(v1[1] - v2[1]) + sqr(v1[2] - v2[2])); }
template<class T> inline T vecSqrDist3(const Vec<T, 3> &v1, const Vec<T, 3> &v2) { return sqr(v1[0] - v2[0]) + sqr(v1[1] - v2[1]) + sqr(v1[2] - v2[2]); }
template<class T> inline T pntSqrDist(const Point_<T> &p1, const Point_<T> &p2) { return sqr(p1.x - p2.x) + sqr(p1.y - p2.y); }
 

 


 

//Mat RC::GetFT(const Mat &img3f)
//{
//	CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
//	Mat sal(img3f.size(), CV_32F), tImg;
//	GaussianBlur(img3f, tImg, Size(3, 3), 0);
//	cvtColor(tImg, tImg, CV_BGR2Lab);
//	Scalar colorM = mean(tImg);
//	for (int r = 0; r < tImg.rows; r++)
//	{
//		float *s = sal.ptr<float>(r);
//		float *lab = tImg.ptr<float>(r);
//		for (int c = 0; c < tImg.cols; c++, lab += 3)
//			s[c] = (float)(sqr(colorM[0] - lab[0]) + sqr(colorM[1] - lab[1]) + sqr(colorM[2] - lab[2]));
//	}
//	normalize(sal, sal, 0, 1, NORM_MINMAX);
//	return sal;
//}
//
//Mat RC::GetHC(const Mat &img3f)
//{
//	// Quantize colors and Á¿»¯ÑÕÉ«
//	Mat idx1i, binColor3f, colorNums1i, weight1f, _colorSal;
//	Quantize(img3f, idx1i, binColor3f, colorNums1i);
//	cvtColor(binColor3f, binColor3f, CV_BGR2Lab);
//
//	normalize(colorNums1i, weight1f, 1, 0, NORM_L1, CV_32F);
//	GetHC(binColor3f, weight1f, _colorSal);
//	float* colorSal = (float*)(_colorSal.data);
//	Mat salHC1f(img3f.size(), CV_32F);
//	for (int r = 0; r < img3f.rows; r++)
//	{
//		float* salV = salHC1f.ptr<float>(r);
//		int* _idx = idx1i.ptr<int>(r);
//		for (int c = 0; c < img3f.cols; c++)
//			salV[c] = colorSal[_idx[c]];
//	}
//	GaussianBlur(salHC1f, salHC1f, Size(3, 3), 0);
//	normalize(salHC1f, salHC1f, 0, 1, NORM_MINMAX);
//	return salHC1f;
//}


int RC::Quantize(const Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio)
{
	static const int clrNums[3] = { 12, 12, 12 };
	static const float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	static const int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

	CV_Assert(img3f.data != NULL);
	idx1i = Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;
	if (img3f.isContinuous() && idx1i.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}

	// Build color pallet
	map<int, int> pallet;
	for (int y = 0; y < rows; y++)
	{
		const float* imgData = img3f.ptr<float>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++, imgData += 3)
		{
			idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
			pallet[idx[x]] ++;
		}
	}

	// Fine significant colors
	int maxNum = 0;
	{
		int count = 0;
		vector<pair<int, int>> num; // (num, color) pairs in num
		num.reserve(pallet.size());
		for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
		sort(num.begin(), num.end(), std::greater< pair<int, int> >());

		maxNum = (int)num.size();
		int maxDropNum = cvRound(rows * cols * (1 - ratio));
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum < 10)
			maxNum = min((int)pallet.size(), 100);
		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i;

		vector<Vec3i> color3i(num.size());
		for (unsigned int i = 0; i < num.size(); i++)
		{
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		for (unsigned int i = maxNum; i < num.size(); i++)
		{
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++)
			{
				int d_ij = vecSqrDist3(color3i[i], color3i[j]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}

	_color3f = Mat::zeros(1, maxNum, CV_32FC3);
	_colorNum = Mat::zeros(_color3f.size(), CV_32S);

	Vec3f* color = (Vec3f*)(_color3f.data);
	int* colorNum = (int*)(_colorNum.data);
	for (int y = 0; y < rows; y++)
	{
		const Vec3f* imgData = img3f.ptr<Vec3f>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			idx[x] = pallet[idx[x]];
			color[idx[x]] += imgData[x];
			colorNum[idx[x]] ++;
		}
	}
	for (int i = 0; i < _color3f.cols; i++)
		color[i] = color[i] / colorNum[i];

	return _color3f.cols;
}
Mat RC::GetRC(const Mat &img3f, double sigmaDist, double segK, int segMinSize, double segSigma)
{
	Mat regIdx1i, colorIdx1i, regSal1v, tmp, _img3f, color3fv;
	if (Quantize(img3f, colorIdx1i, color3fv, tmp) <= 2) // Color quantization
		return Mat::zeros(img3f.size(), CV_32F);
	cvtColor(img3f, _img3f, CV_BGR2Lab);
	cvtColor(color3fv, color3fv, CV_BGR2Lab);
	int regNum = SegmentImage(_img3f, regIdx1i, segSigma, segK, segMinSize);
	vector<Region> regs(regNum);
	BuildRegions(regIdx1i, regs, colorIdx1i, color3fv.cols);
	RegionContrast(regs, color3fv, regSal1v, sigmaDist);
	Mat salimg = Mat::zeros(img3f.size(), CV_32F);
	cv::normalize(regSal1v, regSal1v, 0, 1.0, NORM_MINMAX);
	//cv::normalize(regSal1v, regSal1v, 0, 1.0, NORM_MINMAX, CV_32F);
	double* regSal = regSal1v.ptr<double>(0);
	for (int r = 0; r < img3f.rows; r++){
		const int* regIdx = regIdx1i.ptr<int>(r);
		float* sal = salimg.ptr<float>(r);
		for (int c = 0; c < img3f.cols; c++)
		{
			sal[c] = regSal[regIdx[c]];
		}
	}
	GaussianBlur(salimg, salimg, Size(3, 3), 0);

	return salimg;
}
void RC::calculateSaliencyMap(Mat *src, Mat * dst)
{
	Mat img3f;
	(*src).convertTo(img3f, CV_32FC3, 1.0 / 255);
	*dst= GetRC(img3f, 0.4, 50, 50, 0.5);
}


void RC::BuildRegions(const Mat& regIdx1i, vector<Region> &regs, const Mat &colorIdx1i, int colorNum)
{
	int rows = regIdx1i.rows, cols = regIdx1i.cols, regNum = (int)regs.size();
	Mat_<int> regColorFre1i = Mat_<int>::zeros(regNum, colorNum); // region color frequency
	for (int y = 0; y < rows; y++){
		const int *regIdx = regIdx1i.ptr<int>(y);
		const int *colorIdx = colorIdx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++, regIdx++, colorIdx++){
			Region &reg = regs[*regIdx];
			reg.pixNum++;
			reg.centroid.x += x;
			reg.centroid.y += y;
			regColorFre1i(*regIdx, *colorIdx)++;
		}
	}

	for (int i = 0; i < regNum; i++){
		Region &reg = regs[i];
		reg.centroid.x /= reg.pixNum * cols;
		reg.centroid.y /= reg.pixNum * rows;
		int *regColorFre = regColorFre1i.ptr<int>(i);
		for (int j = 0; j < colorNum; j++){
			float fre = (float)regColorFre[j] / (float)reg.pixNum;
			if (regColorFre[j])
				reg.freIdx.push_back(make_pair(fre, j));
		}
	}
}

void RC::RegionContrast(const vector<Region> &regs, const Mat &color3fv, Mat& regSal1d, double sigmaDist)
{
	Mat_<float> cDistCache1f = Mat::zeros(color3fv.cols, color3fv.cols, CV_32F); {
		Vec3f* pColor = (Vec3f*)color3fv.data;
		for (int i = 0; i < cDistCache1f.rows; i++)
			for (int j = i + 1; j < cDistCache1f.cols; j++)
				cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist3(pColor[i], pColor[j]);
	}

	int regNum = (int)regs.size();
	Mat_<double> rDistCache1d = Mat::zeros(regNum, regNum, CV_64F);
	regSal1d = Mat::zeros(1, regNum, CV_64F);
	double* regSal = (double*)regSal1d.data;
	for (int i = 0; i < regNum; i++){
		for (int j = 0; j < regNum; j++){
			if (i<j) {
				double dd = 0;
				const vector<pair<double, int>> &c1 = regs[i].freIdx, &c2 = regs[j].freIdx;
				for (size_t m = 0; m < c1.size(); m++)
					for (size_t n = 0; n < c2.size(); n++)
						dd += cDistCache1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
				rDistCache1d[j][i] = rDistCache1d[i][j] = dd * exp(-pntSqrDist(regs[i].centroid, regs[j].centroid) / sigmaDist);
			}
			regSal[i] += regs[j].pixNum * rDistCache1d[i][j];
		}
	}
}

 
void RC::SmoothSaliency(const Mat &binColor3f, Mat &sal1d, float delta, const vector<vector<pair<double, int>>> &similar)
{
	if (sal1d.cols < 2)
		return;
	CV_Assert(binColor3f.size() == sal1d.size() && sal1d.rows == 1);
	int binN = binColor3f.cols;
	Vec3f* color = (Vec3f*)(binColor3f.data);
	Mat tmpSal;
	sal1d.copyTo(tmpSal);
	float *sal = (float*)(tmpSal.data);
	float *nSal = (float*)(sal1d.data);

	//* Distance based smooth
	int n = max(cvRound(binN / delta), 2);
	vecF dist(n, 0), val(n);
	for (int i = 0; i < binN; i++)
	{
		const vector<pair<double, int>> &similari = similar[i];
		float totalDist = 0;

		val[0] = sal[i];
		for (int j = 1; j < n; j++)
		{
			int ithIdx = similari[j].second;
			dist[j] = similari[j].first;
			val[j] = sal[ithIdx];
			totalDist += dist[j];
		}
		float valCrnt = 0;
		for (int j = 0; j < n; j++)
			valCrnt += val[j] * (totalDist - dist[j]);

		nSal[i] = valCrnt / ((n - 1) * totalDist);
	}
	//*/

	/* Gaussian smooth
	const float guassCoeff = -0.5f/(delta*delta);
	for (int i = 0; i < binN; i++)
	{
	const vector<pair<double, int>> &similari = similar[i];
	float RCI = sal[i], totalW = 1;

	for (int j = 1; j < binN; j++)
	{
	float w = expf(sqr(similari[j].first)*guassCoeff);
	if (w < 1e-8f)
	break;
	RCI += w * sal[similari[j].second];
	totalW += w;
	}
	nSal[i] = RCI / totalW;
	}
	//*/
}




void RC::AbsAngle(const Mat& cmplx32FC2, Mat& mag32FC1, Mat& ang32FC1)
{
	CV_Assert(cmplx32FC2.type() == CV_32FC2);
	mag32FC1.create(cmplx32FC2.size(), CV_32FC1);
	ang32FC1.create(cmplx32FC2.size(), CV_32FC1);

	for (int y = 0; y < cmplx32FC2.rows; y++)
	{
		const float* cmpD = cmplx32FC2.ptr<float>(y);
		float* dataA = ang32FC1.ptr<float>(y);
		float* dataM = mag32FC1.ptr<float>(y);
		for (int x = 0; x < cmplx32FC2.cols; x++, cmpD += 2)
		{
			dataA[x] = atan2(cmpD[1], cmpD[0]);
			dataM[x] = sqrt(cmpD[0] * cmpD[0] + cmpD[1] * cmpD[1]);
		}
	}
}

void RC::GetCmplx(const Mat& mag32F, const Mat& ang32F, Mat& cmplx32FC2)
{
	CV_Assert(mag32F.type() == CV_32FC1 && ang32F.type() == CV_32FC1 && mag32F.size() == ang32F.size());
	cmplx32FC2.create(mag32F.size(), CV_32FC2);
	for (int y = 0; y < mag32F.rows; y++)
	{
		float* cmpD = cmplx32FC2.ptr<float>(y);
		const float* dataA = ang32F.ptr<float>(y);
		const float* dataM = mag32F.ptr<float>(y);
		for (int x = 0; x < mag32F.cols; x++, cmpD += 2)
		{
			cmpD[0] = dataM[x] * cos(dataA[x]);
			cmpD[1] = dataM[x] * sin(dataA[x]);
		}
	}
}


 