#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <map>
#include <functional>
#include "Segmentation/segment-image.h"
#include <omp.h>
using namespace std;
using namespace cv;
class RC
{
public:
	RC()
	{
	}

	~RC()
	{
	}
	void calculateSaliencyMap(Mat *src, Mat * dst);
	static int Quantize(const Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio = 0.95);
	
	static Mat GetRC(const Mat &img3f, double sigmaDist, double segK, int segMinSize, double segSigma);
private:
	void caRCulateSaliencyMap(Mat *src, Mat * dst);
	static const int SAL_TYPE_NUM = 5;
	 
	static void SmoothSaliency(const Mat &binColor3f, Mat &sal1d, float delta, const vector<vector<pair<double, int>>> &similar);
	static void AbsAngle(const Mat& cmplx32FC2, Mat& mag32FC1, Mat& ang32FC1);
	static void GetCmplx(const Mat& mag32F, const Mat& ang32F, Mat& cmplx32FC2);
	struct Region{
		Region() { pixNum = 0; }
		int pixNum;  // Number of pixels
		vector<pair<double, int> > freIdx;  // Frequency of each color and its index
		Point2d centroid;
	};
	static void BuildRegions(const Mat& regIdx1i, vector<Region> &regs, const Mat &colorIdx1i, int colorNum);
	static void RegionContrast(const vector<Region> &regs, const Mat &color3fv, Mat& regSal1d, double sigmaDist);
};
 