#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
class LC
{
public:
	LC()
	{
	}

	~LC()
	{
	}

public:
	void calculateSaliencyMap(Mat *src, Mat * dst);

};

