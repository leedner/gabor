
//#include"loggabor.h"
//#include"loggabor.cpp"
//
//using namespace loggaborfilter;
//
//void main()
//{
//    Mat img1 = imread("E:\\house.jpg", IMREAD_GRAYSCALE);
//    Mat img2;
//    img1.convertTo(img2, CV_64F, 1.0 / 255.0);
//    int  N = img1.cols;
//    int  M = img1.rows;
//    Mat  filtered_img;
//    garborConvolve(img1, 4, 4, 2.0, 2, 0.3, 40, 1);
//    for (int n = 0; n < 4; n++) {
//      
//    }
//    Mat  X = lowpassfilter(M, N, 0.2, 4);
//    filtered_img = frequency_filter(img2, X);
//    cv::imshow("ԭͼ", img1);
//    cv::imshow("1", filtered_img);
//
//    cv::waitKey(0);
//}