#include "EntropyFil.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
* ind_to_sub
* Convert from linear index to array coordinates.  This is similar
* to MATLAB's IND2SUB function, except that here it's zero-based
* instead of one-based.  This algorithm used here is adapted from
* ind2sub.m.
*
* Inputs
* ======
* p        - zero-based linear index
* num_dims - number of dimensions
* size     - image size (ind_to_sub must be computed with respect
*            to a known image size)
*
* Output
* ======
* coords   - array of array coordinates
*/
void ind_to_sub(int p, int num_dims, const int size[],
	int *cumprod, int *coords)
{
	int j;

	//mxAssert(num_dims > 0, "");
	//mxAssert(coords != NULL, "");
	//mxAssert(cumprod != NULL, "");

	for (j = num_dims - 1; j >= 0; j--)
	{
		coords[j] = p / cumprod[j];
		p = p % cumprod[j];
	}
}

/*
* sub_to_ind
* Convert from array coordinates to linear index.  This is similar
* to MATLAB's SUB2IND function, except that here it's zero-based
* instead of one-based.  The algorithm used here is adapted from
* sub2ind.m.
*
* Inputs
* ======
* coords    - array of array coordinates
* size      - image size (linear index has to be computed with
*             respect to a known image size)
* cumprod   - cumulative product of image size
* num_dims  - number of dimensions
*
* Return
* ======
* zero-based linear index
*/

int sub_to_ind(int *coords, int *cumprod, int num_dims)
{
	int index = 0;
	int k;

	//mxAssert(coords != NULL, "");
	//mxAssert(cumprod != NULL, "");
	//mxAssert(num_dims > 0, "");

	for (k = 0; k < num_dims; k++)
	{
		index += coords[k] * cumprod[k];
	}

	return index;
}

EntropyFilt::EntropyFilt(const cv::Size& winSize, const cv::Size& blockSize)
{
	this->winSize = winSize;
	this->blockSize = blockSize;

	int neighboodNum = winSize.width*winSize.height;
	cv::Size padBlockSize = cv::Size(this->blockSize.width + this->winSize.width - 1, this->blockSize.height + this->winSize.height - 1);

	this->neighboodCorrdsArray = new int[neighboodNum * 2]; //(int*)malloc(sizeof(int)*neighboodNum*neighboodDim);
	this->neighboodOffset = new int[neighboodNum]; //(int*)malloc(sizeof(int)*neighboodNum);
	this->entropyTable = new float[neighboodNum + 1]; //TO DO: change for other window size

	//Contains the cumulative product of the image_size array_;used in the sub_to_ind and ind_to_sub calculations.
	int cumprod[2] = { 1, 1 * winSize.width };
	//cumprod = (int *)malloc(neighboodDim * sizeof(*cumprod));
	//cumprod[0] = 1;
	//cumprod[1] = cumprod[0]*neighboodSize[0];

	int imageCumprod[2] = { 1, 1 * padBlockSize.width }; //= (int*)malloc(2*sizeof(*imageCumprod));

	//initialize neighboodCorrdsArray
	int neighboodSize[2] = { winSize.width, winSize.height };
	for (int p = 0; p < neighboodNum; p++)
	{
		int* coords = this->neighboodCorrdsArray + p * 2;
		ind_to_sub(p, 2, neighboodSize, cumprod, coords);
		//TO DO ind_to_sub(p, neighboodDim, neighboodSize, cumprod, coords);
		for (int q = 0; q < 2; q++)
		{
			coords[q] -= (neighboodSize[q] - 1) / 2;
		}
	}
	//initlalize neighboodOffset in use of neighboodCorrdsArray
	int* elem;
	for (int i = 0; i < neighboodNum; i++)
	{
		elem = this->neighboodCorrdsArray + i * 2;
		this->neighboodOffset[i] = sub_to_ind(elem, imageCumprod, 2);
	}
	//4.calculate entroy for pixel //assuming it is byte TO DO: convert types!!!

	//here,use entropyTable to avoid frequency log function which cost losts of time
	const float log2 = log(2.0f);
	this->entropyTable[0] = 0.0;
	float frequency = 0;
	for (int i = 1; i < neighboodNum + 1; i++)
	{
		frequency = (float)i / neighboodNum;
		this->entropyTable[i] = frequency*(log(frequency) / log2);
	}

}

EntropyFilt::~EntropyFilt()
{
	delete[] neighboodCorrdsArray;
	delete[] neighboodOffset;
	delete[] entropyTable;
}

Mat EntropyFilt::Execute(const Mat& image)
{
	//image must be uint8
	//TO DO: the admin stuff must be done once off

	/*clock_t func_begin,func_end;
	func_begin=clock();*/
	//1.define nerghbood model,here it's 9*9
	//int neighboodDim = 2;
	if ((image.cols != this->blockSize.width) || (image.cols != this->blockSize.width))
		throw string("image must be the same size as blockSize");

	Mat entropyIm(image.rows, image.cols, CV_32FC1);
	entropyIm = 0;

	int neighboodSize[2] = { this->winSize.width, this->winSize.height };

	Mat padImage;
	int left = (neighboodSize[0] - 1) / 2;
	int right = left;
	int top = (neighboodSize[1] - 1) / 2;
	int bottom = top;

	copyMakeBorder(image, padImage, top, bottom, left, right, BORDER_REPLICATE, 0);
	cv::Rect roiRect(top, left, image.cols, image.rows);
	//3.initial neighbood object,reference to Matlab build-in neighbood object system
	int histCount[256] = { 0 };
	int neighboodNum = neighboodSize[0] * neighboodSize[1];
	//4.calculate entroy for pixel //assuming it is byte TO DO: convert types!!!
	uchar* array_ = (uchar*)padImage.data; //NB padImage needs to be continuous for this to work
	int neighboodIndex;
	/*int maxIndex = padImage.cols*padImage.rows;
	float temp;*/
	float entropy;
	int currentIndex = 0;
	int currentIndexInOrigin = 0;
	for (int y = roiRect.y; y < roiRect.height + roiRect.y; y++)
	{
		currentIndex = y*padImage.cols;
		currentIndexInOrigin = (y - roiRect.y)*image.cols;  //-4???
		for (int x = roiRect.x; x < roiRect.width + roiRect.x; x++, currentIndex++, currentIndexInOrigin++)
		{
			for (int j = 0; j < neighboodNum; j++)
			{
				
				neighboodIndex = currentIndex + neighboodOffset[j];
				histCount[array_[neighboodIndex]]++;
			}
			//get entropy
			entropy = 0;
			for (int k = 0; k < 256; k++)
			{
				if (histCount[k] != 0)
				{
					//int frequency = histCount[k];
					entropy -= entropyTable[histCount[k]];
					histCount[k] = 0;
				}
			}
			((float*)entropyIm.data)[currentIndexInOrigin] = entropy;
			
		}
	}
	return entropyIm;
	
}


Mat EntropyFiltFn(const Mat& image)
{
	//image must be uint8	 
	//1.define nerghbood model,here it's 9*9	 
	Mat entropyIm(image.rows, image.cols, CV_32FC1);
	entropyIm = 0;
	int neighboodSize[2] = { 5, 5 };
	//2.Pad gray_src
	Mat padImage;
	int left = (neighboodSize[0] - 1) / 2;
	int right = left;
	int top = (neighboodSize[1] - 1) / 2;
	int bottom = top;
	copyMakeBorder(image, padImage, top, bottom, left, right, BORDER_REPLICATE, 0);	 ;
	cv::Rect roiRect(top, left, image.cols, image.rows);
	//3.initial neighbood object,reference to Matlab build-in neighbood object system
	
	int histCount[256] = { 0 };
	int neighboodNum = neighboodSize[0] * neighboodSize[1];
	/*for(int i = 0; i < neighboodDim; i++)
	neighboodNum *= neighboodSize[i];*/

	//neighboodCorrdsArray is a neighbors_num-by-neighboodDim array_ containing relative offsets
	int* neighboodCorrdsArray = new int[neighboodNum * 2]; //(int*)malloc(sizeof(int)*neighboodNum*neighboodDim);
	//Contains the cumulative product of the image_size array_;used in the sub_to_ind and ind_to_sub calculations.
	int cumprod[2] = { 1, 1 * neighboodSize[0] };
	//cumprod = (int *)malloc(neighboodDim * sizeof(*cumprod));
	//cumprod[0] = 1;
	//cumprod[1] = cumprod[0]*neighboodSize[0];

	int imageCumprod[2] = { 1, padImage.cols }; //= (int*)malloc(2*sizeof(*imageCumprod));
	//imageCumprod[0] = 1;
	//imageCumprod[1] = padImage.cols;

	//initialize neighboodCorrdsArray
	for (int p = 0; p < neighboodNum; p++)
	{
		int* coords = neighboodCorrdsArray + p * 2;
		ind_to_sub(p, 2, neighboodSize, cumprod, coords);
		//TO DO ind_to_sub(p, neighboodDim, neighboodSize, cumprod, coords);
		for (int q = 0; q < 2; q++)
		{
			coords[q] -= (neighboodSize[q] - 1) / 2;
		}
	}
	//initlalize neighboodOffset in use of neighboodCorrdsArray
	int* neighboodOffset = new int[neighboodNum]; //(int*)malloc(sizeof(int)*neighboodNum);
	int* elem;
	for (int i = 0; i < neighboodNum; i++)
	{
		elem = neighboodCorrdsArray + i * 2;
		neighboodOffset[i] = sub_to_ind(elem, imageCumprod, 2);
	}
	//4.calculate entroy for pixel //assuming it is byte TO DO: convert types!!!
	uchar* array_ = (uchar*)padImage.data; //NB padImage needs to be continuous for this to work

	//here,use entropyTable to avoid frequency log function which cost losts of time
	float* entropyTable = new float[neighboodNum + 1]; //TO DO: change for other window size
	const float log2 = log(2.0f);
	entropyTable[0] = 0.0;
	float frequency = 0;
	for (int i = 1; i < neighboodNum + 1; i++)
	{
		frequency = (float)i / neighboodNum;
		entropyTable[i] = frequency*(log(frequency) / log2);
	}

	int neighboodIndex;
	/*int maxIndex = padImage.cols*padImage.rows;
	float temp;*/
	float entropy;
	int currentIndex = 0;
	int currentIndexInOrigin = 0;

	for (int y = roiRect.y; y < roiRect.height + roiRect.y; y++)
	{
		currentIndex = y*padImage.cols;
		currentIndexInOrigin = (y - roiRect.y)*image.cols;  //-4???
		for (int x = roiRect.x; x < roiRect.width + roiRect.x; x++, currentIndex++, currentIndexInOrigin++)
		{
			for (int j = 0; j < neighboodNum; j++)
			{
				//int offset = neighboodOffset[j];
				neighboodIndex = currentIndex + neighboodOffset[j];
				histCount[array_[neighboodIndex]]++;
			}
			//get entropy
			entropy = 0;
			for (int k = 0; k < 256; k++)
			{
				if (histCount[k] != 0)
				{
					//int frequency = histCount[k];
					entropy -= entropyTable[histCount[k]];
					histCount[k] = 0;
				}
			}
			((float*)entropyIm.data)[currentIndexInOrigin] = entropy;
			//((float*)local_entroy_image->imageData)[currentIndexInOrigin] = entropy;
		}
	}
	delete[] neighboodCorrdsArray;
	delete[] neighboodOffset;
	delete[] entropyTable;
	return entropyIm;
	/*func_end=clock();
	double func_time=(double)(func_end-func_begin)/CLOCKS_PER_SEC;
	cout<<"func time"<<func_time<<endl;*/
}