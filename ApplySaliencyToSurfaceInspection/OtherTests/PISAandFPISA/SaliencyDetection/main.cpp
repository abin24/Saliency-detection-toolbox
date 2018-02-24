#include "stdafx.h"
#include "testSingleImage.h"
#include "testImageFolder.h"

#pragma comment( lib, "opencv_core246.lib" )
#pragma comment( lib, "opencv_highgui246.lib" )
#pragma comment( lib, "opencv_imgproc246.lib" )
#pragma comment( lib, "opencv_objdetect246.lib" )
#pragma comment( lib, "CLMF_SDK.lib" )

int main()
{
	clock_t startTime = clock() ;

	//detect Saliency on single image
	//string imgPath = "ImageDatasets\\inputImage.jpg" ;
	//testSingleImg( imgPath ) ;

	//detect saliency on images which are in the given folder.
	string inputFolder = "ImageDatasets\\" ;
	string outputFolder = "ImageDatasets\\out\\" ;

	testImageFolder( inputFolder, outputFolder ) ;

	cout << "Total Time used: " << ( double )( clock() - startTime ) / CLOCKS_PER_SEC << "s." << endl ;

	return 0 ;
}

