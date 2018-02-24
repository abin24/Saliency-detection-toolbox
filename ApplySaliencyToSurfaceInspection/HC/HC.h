#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <map>
#include <functional>

using namespace cv;
using namespace std;


class HC
{
public:
	HC()
	{
	}

	~HC()
	{
	}

public:
	void calculateSaliencyMap(Mat *src, Mat * dst);
private:
	int Quantize(const Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio = 0.95);
	void	GetHC(const Mat &binColor3f, const Mat &weight1f, Mat &_colorSal);
	void SmoothSaliency(const Mat &binColor3f, Mat &sal1d, float delta, const vector<vector< pair<double, int> >> &similar);


};

