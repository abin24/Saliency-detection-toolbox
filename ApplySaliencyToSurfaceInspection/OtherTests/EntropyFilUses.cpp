bool corlor = 0;//When this value is 1, it will be done on the corlor img as it is default
	Mat img = imread("3.jpg");
	imshow("img", img);
	Mat gray;
	cvtColor(img,gray,CV_RGB2GRAY);
	/*Mat Entropymap=  EntropyFiltFn(gray);
	imshow("Entropymap", Entropymap / 5.2);*/
	resize(gray,gray,Size(gray.cols/4,gray.rows/4));
	EntropyFilt ef(Size(9, 9), gray.size());
	Mat Entropymap2 = ef.Execute(gray);
	 
	imshow("Entropymap2", Entropymap2 / 5.2);