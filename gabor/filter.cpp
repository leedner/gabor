//#include"loggabor.cpp"
//#include"loggabor.h"
//
//using namespace loggaborfilter; 
//
//void main()
//{
//	Mat img1 = imread("E:\\house.jpg", IMREAD_GRAYSCALE);
//	int channels = img1.channels();
//	Mat img2;
//	img1.convertTo(img2, CV_64F, 1.0 / 255.0);
//	int  N = img1.cols;
//	int  M = img1.rows;
//	Mat  filtered_img;
//	GaborConvolveResult* prt_result = new GaborConvolveResult;
//	loggarborConvolve(prt_result, img1, 4, 4, 3.0, 1.7, 0.3, 1.3);
//	for (int n = 0; n < 4; n++) {
//		imshow("vector", prt_result->BP[n]);
//		imshow("vector1", prt_result->EO[n][n]);
//	}
//	Mat  X = lowpassfilter(M, N, 0.2, 4);
//	filtered_img = frequency_filter(img2, X);
//	cv::imshow("yauntu", img1);
//	cv::imshow("1", filtered_img);
//	cv::waitKey(0);
//}