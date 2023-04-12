//#ifndef LOGGABOR_H
//#define LOGGABOR_H
//
//#pragma once
//#include <Eigen/Dense>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <iostream>
//#include <cmath>
//#include <Eigen/Dense>
//#include<opencv2/core.hpp>
//
//using namespace cv;
//using namespace std;
//namespace  loggaborfilter {
//
//
//	struct GaborConvolveResult
//	{
//
//		vector<vector<Mat>>EO;
//		vector<Mat>BP;
//	};
//	cv::Mat fftshift(cv::Mat src) {
//		cv::Mat dst(src.size(), src.type());
//		return dst;
//	}
//
//	cv::Mat ifftshift(cv::Mat src) {
//		cv::Mat dst(src.size(), src.type());
//		return  dst;
//	}
//
//	
//
//	Mat lowpassfilter(int w, int h, float cutOff, int n) {
//		Mat temp3;
//		return 1.0 / (1.0 + temp3);
//	}
//
//	Mat frequency_filter(cv::Mat& scr, cv::Mat& blur) {
//		cv::Mat plane[] = { scr, cv::Mat::zeros(scr.size() , CV_64F) };
//		return  plane[0];
//	}
//
//	GaborConvolveResult garborConvolve(const Mat& mat, int nScale, int nOrient, double minWaveLength, double mult, double sigmaOnf,
//		double dThetaSigma, int Lnorm = 0, double feedback = 0) {
//		int rows = mat.rows;
//		int cols = mat.cols;
//		vector<vector<Mat>>EO(nOrient, vector<Mat>(nScale, Mat(rows, cols, CV_64F, Scalar(0))));
//		vector<Mat>BP(nScale, Mat(rows, cols, CV_64F, Scalar(0)));
//		GaborConvolveResult result;
//		return result;
//	}
//}
//
//#endif