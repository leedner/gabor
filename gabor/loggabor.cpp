#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include<opencv2/core.hpp>
#include"loggabor.h"

using namespace cv;
using namespace std;


	struct GaborConvolveResult
	{

		vector<vector<Mat>>EO;
		vector<Mat>BP;
	};


	cv::Mat fftshift(cv::Mat src)
	{
		cv::Mat dst(src.size(), src.type());

		int x0 = src.cols / 2;
		int x1 = src.cols - x0;
		int y0 = src.rows / 2;
		int y1 = src.rows - y0;

		cv::Mat src_b0(src, cv::Rect(0, 0, x1, y1));
		cv::Mat src_b1(src, cv::Rect(x1, 0, x0, y1));
		cv::Mat src_b2(src, cv::Rect(0, y1, x1, y0));
		cv::Mat src_b3(src, cv::Rect(x1, y1, x0, y0));
		cv::Mat dst_b0(dst, cv::Rect(0, 0, x0, y0));
		cv::Mat dst_b1(dst, cv::Rect(x0, 0, x1, y0));
		cv::Mat dst_b2(dst, cv::Rect(0, y0, x0, y1));
		cv::Mat dst_b3(dst, cv::Rect(x0, y0, x1, y1));

		src_b1.copyTo(dst_b2);
		src_b2.copyTo(dst_b1);
		src_b0.copyTo(dst_b3);
		src_b3.copyTo(dst_b0);
		return dst;
	}


	cv::Mat ifftshift(cv::Mat src)
	{
		cv::Mat dst(src.size(), src.type());

		int x1 = src.cols / 2;
		int x0 = src.cols - x1;
		int y1 = src.rows / 2;
		int y0 = src.rows - y1;

		cv::Mat src_b0(src, cv::Rect(0, 0, x1, y1));
		cv::Mat src_b1(src, cv::Rect(x1, 0, x0, y1));
		cv::Mat src_b2(src, cv::Rect(0, y1, x1, y0));
		cv::Mat src_b3(src, cv::Rect(x1, y1, x0, y0));
		cv::Mat dst_b0(dst, cv::Rect(0, 0, x0, y0));
		cv::Mat dst_b1(dst, cv::Rect(x0, 0, x1, y0));
		cv::Mat dst_b2(dst, cv::Rect(0, y0, x0, y1));
		cv::Mat dst_b3(dst, cv::Rect(x0, y0, x1, y1));

		src_b1.copyTo(dst_b2);
		src_b2.copyTo(dst_b1);
		src_b0.copyTo(dst_b3);
		src_b3.copyTo(dst_b0);
		return dst;
	}


	Mat lowpassfilter(int w, int h, float cutOff, int n) {


		Eigen::VectorXd xrange = Eigen::VectorXd::LinSpaced(w, -0.5, 0.5);
		Eigen::VectorXd yrange = Eigen::VectorXd::LinSpaced(h, -0.5, 0.5);

		Eigen::MatrixXd x(w, h), y(w, h);
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j < h; ++j) {
				x(i, j) = xrange(i);
				y(i, j) = yrange(j);
			}
		}
		Mat  X, Y, r;
		cv::eigen2cv(x, X);
		cv::eigen2cv(y, Y);
		Mat temp1, temp2,temp3;
		cv::pow(X, 2.0, temp1);
		cv::pow(Y, 2.0, temp2);
		sqrt(temp1 + temp2, r);
		Mat r1 = ifftshift(r);
		pow((r1 / cutOff), static_cast<double>(2 * n), temp3);
		return 1.0 / (1.0 + temp3);
	}

	Mat frequency_filter(cv::Mat& scr, cv::Mat& blur)
	{
		//std::cout << scr.type() <<endl;
		cv::Mat plane[] = { scr, cv::Mat::zeros(scr.size() , CV_64F) };
		//std::cout << plane[0].type() << endl;
		//std::cout << plane[1].type() << endl;
		cv::Mat complexIm;
		//std::cout << complexIm.channels()<<endl;
		cv::merge(plane, 2, complexIm);
		int  cn = complexIm.channels();
		cv::dft(complexIm, complexIm);
		int n = complexIm.channels();
		// 分离通道（数组分离）

		cv::split(complexIm, plane);
		// 以下的操作是频域迁移
		fftshift(plane[0]);
		fftshift(plane[1]);

		// *****************滤波器函数与DFT结果的乘积****************
		cv::Mat blur_r, blur_i, BLUR;
		cv::multiply(plane[0], blur, blur_r, 1.0, 5);
		cv::multiply(plane[1], blur, blur_i, 1.0, 5);
		cv::Mat plane1[] = { blur_r, blur_i };
		cv::imshow("plane0", blur_r);
		cv::waitKey(0);
		cv::imshow("plane1", blur_i);
		cv::waitKey(0);


		// 再次搬移回来进行逆变换
		fftshift(plane1[0]);
		fftshift(plane1[1]);
		cv::merge(plane1, 2, BLUR); // 实部与虚部合并
		cv::idft(BLUR, BLUR);       // idft结果也为复数
		BLUR = BLUR / BLUR.rows / BLUR.cols;
		cv::split(BLUR, plane);//分离通道，主要获取通道
		return plane[0];
	}


	GaborConvolveResult garborConvolve(const Mat& mat, int nScale, int nOrient, double minWaveLength, double mult, double sigmaOnf,
		double dThetaSigma, int Lnorm = 0, double feedback = 0) {
		int rows = mat.rows;
		int cols = mat.cols;
		Mat  matdft(cols, rows, CV_64F);
		matdft = fftshift(mat);

		vector<vector<Mat>>EO(nOrient, vector<Mat>(nScale, Mat(rows, cols, CV_64F, Scalar(0))));
		vector<Mat>BP(nScale, Mat(rows, cols, CV_64F, Scalar(0)));

		Eigen::VectorXd xrange = Eigen::VectorXd::LinSpaced(rows, -0.5, 0.5);//cols =511
		Eigen::VectorXd yrange = Eigen::VectorXd::LinSpaced(cols, -0.5, 0.5);//rows =473

		Eigen::MatrixXd x(rows, cols), y(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				x(i, j) = xrange(i);
				y(i, j) = yrange(j);
			}
		}
		//std::cout << x.rows()<<endl<<x.cols();
		Mat  X, Y, r;
		cv::eigen2cv(x, X);
		cv::eigen2cv(y, Y);
		Mat temp1, temp2, temp3;
		cv::pow(X, 2.0, temp1);
		cv::pow(Y, 2.0, temp2);
		sqrt(temp1 + temp2, r);

		Mat theta(X.rows, X.cols, CV_64F);
		//std::cout << Y.type() << endl;;
		for (int i = 0; i < X.rows; i++)
			for (int j = 0; j < X.cols; j++)
				theta.at<double>(i, j) = atan2(Y.at<double>(i, j), X.at<double>(i, j));
		fftshift(r);
		fftshift(theta);
		r.at<double>(0, 0) = 1.0;
		Mat sinTheta(X.rows, X.cols, CV_64F);
		Mat cosTheta(X.rows, X.cols, CV_64F);

		for (int i = 0; i < X.rows; ++i)
			for (int j = 0; j < X.cols; ++j)
				sinTheta.at<double>(i, j) = sin(theta.at<double>(i, j));

		for (int i = 0; i < X.rows; ++i)
			for (int j = 0; j < X.cols; ++j)
				cosTheta.at<double>(i, j) = cos(theta.at<double>(i, j));

		Mat lp = lowpassfilter(rows, cols, 0.2, 4);
		vector<Mat>logGabor;
		for (int s = 0; s < nScale; ++s)
		{
			logGabor.push_back(Mat(rows, cols, CV_64F));
			double waveLength = minWaveLength * pow(mult, s);
			double fo = 1.0 / waveLength;

			Mat tempUpper;
			log(r / fo, tempUpper);
			pow(tempUpper, 2, tempUpper);

			double tempLower = pow(log(sigmaOnf), 2);

			double factory = -1 / 2.0;
			tempUpper = tempUpper / tempLower * factory;
			exp(tempUpper, logGabor[s]);

			logGabor[s] = logGabor[s].mul(lp);
			logGabor[s].at<double>(0, 0) = 0;

			double L = 0;
			switch (Lnorm)
			{
			case 0:
				L = 1;
				break;
			case 1:
			{
				Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
				Mat complex;
				cv::dft(logGabor[s], complex, DFT_COMPLEX_OUTPUT + DFT_SCALE);
				split(complex, planes);
				Mat realPart = planes[0];

				L = sum(abs(realPart))[0];
				break;
			}

			case 2:
			{
				Mat temp;
				pow(logGabor[s], 2, temp);

				L = sqrt(sum(temp)[0]);

			}

			break;
			default:
				break;
			}

			logGabor[s] = logGabor[s] / L;
			cout << logGabor[s].type() << endl;
			//cout << curLogGabor;
			Mat matdft_64F;
			matdft.convertTo(matdft_64F, CV_64F, 1.0 / 255.0);
			BP[s]=frequency_filter(matdft_64F, logGabor[s]);
			imshow("BP", BP[s]);
			waitKey(0);
			/*Mat complex;
			cv::Mat planes[] = { matdft_64F, cv::Mat::zeros(matdft.size() , CV_64F) };
			split(matdft_64F, planes);
		
			planes[0] = planes[0].mul(logGabor[s]);
			planes[1] = planes[1].mul(logGabor[s]);

			Mat complexd;
			merge(planes, 2, complexd);
			cv::dft(complexd, BP[s], DFT_COMPLEX_OUTPUT + DFT_SCALE);*/			
		}
		//std::cout << logGabor[0] << endl;

		for (int o = 0; o < nOrient; ++o)
		{
			double angl = o * CV_PI / nOrient;
			double waveLength = minWaveLength;


			Mat ds = sinTheta * cos(angl) - cosTheta * sin(angl);
			Mat dc = cosTheta * cos(angl) + sinTheta * sin(angl);

			Mat dTheta(rows, cols, CV_64F);
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					dTheta.at<double>(i, j) = abs(atan2(ds.at<double>(i, j), dc.at<double>(i, j)));

			Mat temp;
			pow(dTheta, 2, temp);
			temp = -temp;
			Mat spread;
			double thetaSigma = CV_PI / nOrient / dThetaSigma;
			exp(temp / (2 * pow(thetaSigma, 2)), spread);

			for (int s = 0; s < nScale; ++s)
			{
				Mat filter = spread.mul(logGabor[s]);
				double L = 0;
				switch (Lnorm)
				{
				case 0: L = 1;
					break;
				case 1:
				{
					Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
					Mat complex;
					cv::dft(filter, complex, DFT_COMPLEX_OUTPUT + DFT_SCALE);
					split(complex, planes);
					Mat realPart = planes[0];
					L = sum(abs(realPart))[0];
				}
				break;
				case 2:
				{


					Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };

					split(temp, planes);
					Mat realPart = planes[0];


					Mat imagPart = planes[1];
					pow(realPart, 2, realPart);
					pow(imagPart, 2, imagPart);


					L = sqrt(sum(realPart)[0] + sum(imagPart)[0]);
				}
				break;
				default:
					break;
				}
				filter = filter / L;
				Mat complex;
				Mat matdft_64F;
				matdft.convertTo(matdft_64F, CV_64F, 1.0 / 255.0);
				EO[o][s]= frequency_filter(matdft_64F, filter);
				//Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
				//cv::split(matdft_64F, planes);

				//planes[0] = planes[0].mul(filter);
				//planes[1] = planes[1].mul(filter);

				//merge(planes, 2, complex);

				////here
				////Mat  multed = matDft.mul(filter);
				////cout << filter << endl;
				//cv::dft(complex, EO[o][s], DFT_COMPLEX_OUTPUT + DFT_SCALE);
				////cout << EO[s][o].cols << " " << EO[s][o].rows << EO[s][o].channels() << " " << EO[s][o].depth() << endl;

				//Mat EOPlanes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
				//split(EO[o][s], EOPlanes);
				waveLength = waveLength * mult;
			}


		}
		GaborConvolveResult result;
		result.BP = BP;
		cout << BP[0].channels();
		result.EO = EO;
		return result;
	}



void main()
{
	Mat img1 = imread("E:\\house.jpg", IMREAD_GRAYSCALE);
	int channels = img1.channels();
	Mat img2;
	img1.convertTo(img2, CV_64F, 1.0 / 255.0);
	int  N = img1.cols;
	int  M = img1.rows;
	Mat  filtered_img;
	for (int n = 0; n < 4; n++) 
		imshow("vector", garborConvolve(img1, 4, 4, 2.0, 2, 0.3, 40, 1).BP[n]);
	Mat  X = lowpassfilter(M, N, 0.2, 4);
	filtered_img = frequency_filter(img2, X);
	cv::imshow("yauntu", img1);
	cv::imshow("1", filtered_img);
	cv::waitKey(0);
}