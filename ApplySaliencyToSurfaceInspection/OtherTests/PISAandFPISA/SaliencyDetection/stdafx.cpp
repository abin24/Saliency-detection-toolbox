#include "stdafx.h"
#include<functional>

/*
	Get the list of image names and numbers of images of the given image folder
	Parameters:
		Output:
			fileNames: list of image names
			fileCount: Number of images
		Input:
			imgFolder: the path of the image folder
	Implement Details:
		Implement by using Windows API
	Warning:
		No sub-folders allowed in the given image folder
*/
void getFileNames( string fileNames[], int& fileCount, string imgFolder, string subFolderPath )
{
	struct _finddata_t filefind ;
	intptr_t handle ;
	string foldPath = imgFolder + "*.*" ;

	if ( ( handle = _findfirst( foldPath.c_str(), &filefind ) ) == -1L ) return ;

	do 
	{
		if ( filefind.attrib & _A_SUBDIR)    
		{
			if( (strcmp(filefind.name,".") != 0 ) &&(strcmp(filefind.name,"..") != 0))   
			{
				subFolderPath = filefind.name ; 
				subFolderPath += "\\" ;
				string newPath = imgFolder + subFolderPath ;
				getFileNames( fileNames, fileCount, newPath, subFolderPath );
			}
		}
		else
		{
			fileNames[fileCount++] = subFolderPath + filefind.name ;
		}		
	}
	while( 0 == _findnext( handle, &filefind ) ) ;
	_findclose( handle ) ;  
}

/*
 Normalize the image via sigmoid-like function.
 Parameters:
	Input:
		srcSal: input data
		halfRange: the domain of the sigmoid-like function
	Output: 
		dstSal: normalized results
*/
void normalizeBySigmoid( const Mat& srcSal, Mat& dstSal, int halfRange )
{
	normalize( srcSal, dstSal, -halfRange, halfRange, NORM_MINMAX ) ;
	
	exp( -dstSal, dstSal ) ;
	dstSal = 1 / ( 1 + dstSal ) ;
}

/*
	Combine color and structure contrast measures together to get final saliency result by using additive model
	Parameters:
		Input:
			strucSal: saliency map of structure contrast measure
			colorSal: saliency map of color contrast measure
			strucClusterIndx: index map of structure contrast measure
			colorClusterIndx: index map of color contrast measure
			strucClusterRank: rank map of structure contrast measure
			colorClusterRank: rank map of color contrast measure
			strucClusterVar: variance map of structure contrast measure
			colorClusterVar: variance map of color contrast measure
			thresh: threshold for descideSaliency
		Output:
			sal: final saliency result
*/
void descideSaliency( Mat& sal, 
	const Mat& strucSal, const Mat& colorSal, 
	const Mat& strucClusterIndx, const Mat& colorClusterIndx, 
	const Mat& strucClusterRank, const Mat& colorClusterRank, 
	const Mat& strucClusterVar, const Mat& colorClusterVar, 
	int thresh
	)
{
	sal = Mat::zeros( strucSal.size(), CV_32FC1 ) ;

	const float* colorRank = colorClusterRank.ptr<float>( 0 ) ;
	const float* strucRank = strucClusterRank.ptr<float>( 0 ) ;
	const float* colorVar = colorClusterVar.ptr<float>( 0 ) ;
	const float* strucVar = strucClusterVar.ptr<float>( 0 ) ;
	for( int row = 0; row < sal.rows; row++ )
	{
		float* salPtr = sal.ptr<float>( row ) ;
		const float* strucSalPtr = strucSal.ptr<float>( row ) ;
		const float* colorSalPtr = colorSal.ptr<float>( row ) ;
		const int* colorIndx = colorClusterIndx.ptr<int>( row ) ;
		const int* strucIndx = strucClusterIndx.ptr<int>( row ) ;		
		for( int col = 0; col < sal.cols; col++ )
		{
			if ( strucRank[strucIndx[col]] <= thresh || colorRank[colorIndx[col]] <= thresh )
			{
				salPtr[col] = (strucSalPtr[col] + colorSalPtr[col]) / 2.0f ;
			}
			else if ( strucVar[strucIndx[col]] > colorVar[colorIndx[col]] )
			{
				salPtr[col] = colorSalPtr[col] ;
			}
		}
	}
}


/* Decide cluster number K by choosing most frequently occurring features by ensuring they cover XX% of histogram distributions,
and the feature is defined by the combination of two histograms.
	Parameters:
		Input:
			pallet: frequency of each feature
			featurnNum: total feature number
			ratio: XX%, default is 95%
		Output:
			return cluster number K
*/
int getFrequentFeatureNum( map<int, int>& pallet, int featureNum, float ratio )
{
	int significantNum = 0;

	vector<pair<int, int>> palletTranspose; 
	palletTranspose.reserve(pallet.size());
	for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
		palletTranspose.push_back(pair<int, int>(it->second, it->first)); 
	sort(palletTranspose.begin(), palletTranspose.end(), std::greater<pair<int, int>>());

	significantNum = (int)palletTranspose.size();
	int maxDropNum = cvRound( featureNum * ( 1 - ratio ) );
	for (int i = palletTranspose[significantNum - 1].first; i < maxDropNum && significantNum > 1; significantNum--)
		i += palletTranspose[significantNum - 2].first;

	return std::max( significantNum, 3 )  ;
}