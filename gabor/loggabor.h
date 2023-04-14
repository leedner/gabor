#ifndef LOGGABOR_H
#define LOGGABOR_H


#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include<opencv2/core.hpp>

using namespace cv;
using namespace std;
namespace  loggaborfilter {

	struct GaborConvolveResult
	{
		vector<vector<Mat>>EO;
		vector<Mat>BP;
	};

	vector<Mat> bank;

	cv::Mat fftshift(cv::Mat src);


	cv::Mat ifftshift(cv::Mat src);


	Mat lowpassfilter(int w, int h, float cutOff, int n);


	void loggarborConvolve(GaborConvolveResult* ptr, const Mat& mat, int nScale, int nOrient, double minWaveLength, double mult, double sigmaOnf,
		double dThetaSigma, int Lnorm , double feedback );

	void filter_make(Mat scr, int  nScale, int nOrient, GaborConvolveResult* pbank);
}

#endif