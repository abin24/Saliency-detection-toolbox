#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;  
class FT
{
public:
	FT()
	{
	}

	~FT()
	{
	}

public:
	static void calculateSaliencyMap(Mat *src, Mat * dst, bool corlor = 1,int ksize=3);
	 
};
 
 