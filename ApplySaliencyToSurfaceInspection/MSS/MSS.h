#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
class MSS
{
public:
	MSS();
	~MSS();

public:
	void calculateSaliencyMap(Mat *src, Mat * dst);

};
