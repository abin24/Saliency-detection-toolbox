#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
class PHOT
{
public:
	PHOT();
	~PHOT();

public:
	static void calculateSaliencyMap(Mat *src, Mat * dst);

};
