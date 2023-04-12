//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <cmath>
//
//using namespace cv;
//using namespace std;
//
//void meshgrid(double xStart, double xEnd, double yStart, double yEnd, Mat& X, Mat& Y)
//{
//	std::vector<double> t_x, t_y;
//	while (xStart <= xEnd)
//	{
//		t_x.push_back(xStart);
//		++xStart;
//	}
//	while (yStart <= yEnd)
//	{
//		t_y.push_back(yStart);
//		++yStart;
//	}
//	repeat(t_x, t_y.size(), 1, X);//强制转换出问题了
//	repeat(t_x, t_y.size(), 1, Y);//t-y 写错了
//}
//
//
//void fftshift(cv::Mat& plane0, cv::Mat& plane1)
//{
//	// 以下的操作是移动图像  (零频移到中心)
//	int cx = plane0.cols / 2;
//	int cy = plane0.rows / 2;
//	cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // 元素坐标表示为(cx, cy)
//	cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));
//
//	cv::Mat temp;
//	part1_r.copyTo(temp);  //左上与右下交换位置(实部)
//	part4_r.copyTo(part1_r);
//	temp.copyTo(part4_r);
//
//	part2_r.copyTo(temp);  //右上与左下交换位置(实部)
//	part3_r.copyTo(part2_r);
//	temp.copyTo(part3_r);
//
//	cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
//	cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));
//
//	part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
//	part4_i.copyTo(part1_i);
//	temp.copyTo(part4_i);
//
//	part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
//	part3_i.copyTo(part2_i);
//	temp.copyTo(part3_i);
//}
//
//Mat lowpassfilter_kernel(Mat& x, Mat& y, double cutOff,int n)
//{
//	Mat radius;
//	Mat x2;
//	Mat y2;
//	pow(x, 2, x2);
//	pow(y, 2, y2);
//	sqrt(x2 + y2, radius);
//	Mat f;
//	Mat temp;
//	pow((radius / cutOff), 2 * n, temp);
//	for (int i=0;i<)
//	return  f;
//}
//
//Mat lowpassFilter(Mat&scr,Size size, double cutOff, int n)
//{
//	int rows, cols;
//	double xStart, xEnd, yStart, yEnd;//xstart an  ystart  is the range of the pic
//	if (cutOff < 0 || cutOff > 0.5)
//		throw std::exception("cutoff frequency must be between 0 and 0.5");
//
//	if (size.width == 1)
//	{
//		rows = size.width;
//		cols = size.width;
//	}
//	else
//	{
//		rows = size.height;
//		cols = size.width;
//	}
//
//	Mat x(cols, rows, CV_64FC1);
//	Mat y(cols, rows, CV_64FC1);
//
//	bool isRowsOdd = (rows % 2 == 1);
//	bool isColsOdd = (cols % 2 == 1);
//	if (isRowsOdd)
//	{
//		yStart = -(rows - 1) / 2.0;
//		yEnd = (rows - 1) / 2.0;
//	}
//	else
//	{
//		yStart = -(rows / 2.0);
//		yEnd = (rows / 2.0 - 1);
//	}
//
//	if (isColsOdd)
//	{
//		xStart = -(cols - 1) / 2.0;
//		xEnd = (cols - 1) / 2.0;
//	}
//	else
//	{
//		xStart = -(cols / 2.0);
//		xEnd = (cols / 2.0 - 1);
//	}
//	meshgrid(xStart, xEnd, yStart, yEnd, x, y);//   bug here
//	if (isColsOdd)
//		x /= (cols - 1);
//	else
//		x /= cols;
//
//	if (isRowsOdd)
//		y /= (rows - 1);
//	else
//		y /= rows;
//	Mat blur =lowpassfilter_kernel(x, y, cutOff,n);
//	Mat plane[] = { Mat_<float>(scr),Mat::zeros(scr.size(),CV_32F) };
//	Mat complexIm;
//	merge(plane, 2, complexIm);
//	dft(complexIm, complexIm);
//	fftshift(plane[0], plane[1]);
//	Mat blur_r, blur_i, BLUR;
//	cv::multiply(plane[0], blur, blur_r);  // 滤波（实部与滤波器模板对应元素相乘）
//	cv::multiply(plane[1], blur, blur_i);  // 滤波（虚部与滤波器模板对应元素相乘）
//	cv::Mat plane1[] = { blur_r, blur_i };
//
//	// 再次搬移回来进行逆变换
//	fftshift(plane1[0], plane1[1]);
//	cv::merge(plane1, 2, BLUR); // 实部与虚部合并
//	cv::idft(BLUR, BLUR);       // idft结果也为复数
//	BLUR = BLUR / BLUR.rows / BLUR.cols;
//	cv::split(BLUR, plane);//分离通道，主要获取通道
//	return plane[0];
//}
//
//
//struct GaborConvolveResult
//{
//
//	vector<vector<Mat>>EO;
//	vector<Mat>BP;
//};
//
//
//GaborConvolveResult garborConvolve(const Mat& mat, int nScale, int nOrient, double minWaveLength, double mult, double sigmaOnf,
//	double dThetaSigma, int Lnorm = 0, double feedback = 0)
//{
//	//mat应已确认是灰度图,并且是double
//	int rows = mat.rows;
//	int cols = mat.cols;
//
//	Mat matDft;
//	matDft= fftshift(mat);
//
//	vector<vector<Mat>>EO(nOrient, vector<Mat>(nScale, Mat(cols, rows, CV_64F, Scalar(0))));
//	vector<Mat>BP(nScale, Mat(cols, rows, CV_64F, Scalar(0)));
//
//	Mat x;
//	Mat y;
//	double xStart, xEnd, yStart, yEnd;
//	bool isRowsOdd = (rows % 2 == 1);
//	bool isColsOdd = (cols % 2 == 1);
//	if (isRowsOdd)
//	{
//		yStart = -(rows - 1) / 2.0;
//		yEnd = (rows - 1) / 2.0;
//	}
//	else
//	{
//		yStart = -(rows / 2.0);
//		yEnd = (rows / 2.0 - 1);
//	}
//
//	if (isColsOdd)
//	{
//		xStart = -(cols - 1) / 2.0;
//		xEnd = (cols - 1) / 2.0;
//	}
//	else
//	{
//		xStart = -(cols / 2.0);
//		xEnd = (cols / 2.0 - 1);
//	}
//
//
//
//
//	meshgrid(xStart, xEnd, yStart, yEnd, x, y);
//
//	if (isColsOdd)
//		x /= (cols - 1);
//	else
//		x /= cols;
//
//	if (isRowsOdd)
//		y /= (rows - 1);
//	else
//		y /= rows;
//
//	Mat radius;
//	Mat x2;
//	Mat y2;
//
//
//	pow(x, 2, x2);
//	pow(y, 2, y2);
//	sqrt(x2 + y2, radius);
//
//	Mat theta(rows, cols, CV_64FC1);
//	//求出每个位置对应的theta
//	for (int i = 0; i < rows; ++i)
//		for (int j = 0; j < cols; ++j)
//			theta.at<double>(i, j) = atan2(y.at<double>(i, j), x.at<double>(i, j));
//
//	fftshift(radius);
//	fftshift(theta);
//
//	radius.at<double>(0, 0) = 1;
//	//求出每个位置的cos和sin
//	Mat sinTheta(rows, cols, CV_64FC1);
//	Mat cosTheta(rows, cols, CV_64FC1);
//
//	for (int i = 0; i < rows; ++i)
//		for (int j = 0; j < cols; ++j)
//			sinTheta.at<double>(i, j) = sin(theta.at<double>(i, j));
//
//	for (int i = 0; i < rows; ++i)
//		for (int j = 0; j < cols; ++j)
//			cosTheta.at<double>(i, j) = cos(theta.at<double>(i, j));
//
//	Mat lp = lowpassFilter(Size(cols, rows), 0.45, 15);
//
//
//
//	//vector<Mat>logGabor(nScale, Mat(rows, cols, CV_64F,Scalar(0,0)));
//	vector<Mat>logGabor;
//	for (int s = 0; s < nScale; ++s)
//	{
//		logGabor.push_back(Mat(rows, cols, CV_64F));
//		double waveLength = minWaveLength * pow(mult, s);
//		double fo = 1.0 / waveLength;
//
//		Mat tempUpper;
//		log(radius / fo, tempUpper);
//		pow(tempUpper, 2, tempUpper);
//
//		double tempLower = pow(log(sigmaOnf), 2);
//
//		double factory = -1 / 2.0;
//		tempUpper = tempUpper / tempLower * factory;
//		exp(tempUpper, logGabor[s]);
//
//		logGabor[s] = logGabor[s].mul(lp);
//		logGabor[s].at<double>(0, 0) = 0;
//
//		double L = 0;
//		switch (Lnorm)
//		{
//		case 0:
//			L = 1;
//			break;
//		case 1:
//		{
//			Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
//			Mat complex;
//			idft(logGabor[s], complex, DFT_COMPLEX_OUTPUT + DFT_SCALE);
//			split(complex, planes);
//			Mat realPart = planes[0];
//
//			L = sum(abs(realPart))[0];
//			break;
//		}
//
//		case 2:
//		{
//			Mat temp;
//			pow(logGabor[s], 2, temp);
//
//			L = sqrt(sum(temp)[0]);
//
//		}
//
//		break;
//		default:
//			break;
//		}
//
//		logGabor[s] = logGabor[s] / L;
//		//cout << logGabor[s] << endl;
//		//cout << curLogGabor;
//
//
//		Mat complex;
//		Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
//		split(matDft, planes);
//
//		planes[0] = planes[0].mul(logGabor[s]);
//		planes[1] = planes[1].mul(logGabor[s]);
//
//		//idft(complex, EO, DFT_COMPLEX_OUTPUT + DFT_SCALE);
//
//
//		//cout << temp.depth() << "  " << temp.channels();
//		//split(temp, planes);
//
//		Mat complexd;
//		merge(planes, 2, complexd);
//		idft(complexd, BP[s], DFT_COMPLEX_OUTPUT + DFT_SCALE);
//
//	}
//	cout << logGabor[0] << endl;
//
//	for (int o = 0; o < nOrient; ++o)
//	{
//		double angl = o * CV_PI / nOrient;
//		double waveLength = minWaveLength;
//
//
//		Mat ds = sinTheta * cos(angl) - cosTheta * sin(angl);
//		Mat dc = cosTheta * cos(angl) + sinTheta * sin(angl);
//
//		Mat dTheta(rows, cols, CV_64F);
//		for (int i = 0; i < rows; ++i)
//			for (int j = 0; j < cols; ++j)
//				dTheta.at<double>(i, j) = abs(atan2(ds.at<double>(i, j), dc.at<double>(i, j)));
//
//		Mat temp;
//		pow(dTheta, 2, temp);
//		temp = -temp;
//		Mat spread;
//		double thetaSigma = CV_PI / nOrient / dThetaSigma;
//		exp(temp / (2 * pow(thetaSigma, 2)), spread);
//
//		for (int s = 0; s < nScale; ++s)
//		{
//			Mat filter = spread.mul(logGabor[s]);
//			double L = 0;
//			switch (Lnorm)
//			{
//			case 0: L = 1;
//				break;
//			case 1:
//			{
//				Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
//				Mat complex;
//				idft(filter, complex, DFT_COMPLEX_OUTPUT + DFT_SCALE);
//				split(complex, planes);
//				Mat realPart = planes[0];
//				L = sum(abs(realPart))[0];
//			}
//			break;
//			case 2:
//			{
//
//
//				Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
//
//				split(temp, planes);
//				Mat realPart = planes[0];
//
//
//				Mat imagPart = planes[1];
//				pow(realPart, 2, realPart);
//				pow(imagPart, 2, imagPart);
//
//
//				L = sqrt(sum(realPart)[0] + sum(imagPart)[0]);
//			}
//			break;
//			default:
//				break;
//			}
//			filter = filter / L;
//
//			Mat complex;
//			Mat planes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
//			cv::split(matDft, planes);
//
//			planes[0] = planes[0].mul(filter);
//			planes[1] = planes[1].mul(filter);
//
//			merge(planes, 2, complex);
//
//			//here
//			//Mat  multed = matDft.mul(filter);
//			//cout << filter << endl;
//			idft(complex, EO[o][s], DFT_COMPLEX_OUTPUT + DFT_SCALE);
//			//cout << EO[s][o].cols << " " << EO[s][o].rows << EO[s][o].channels() << " " << EO[s][o].depth() << endl;
//
//			Mat EOPlanes[2] = { Mat(rows,cols,CV_64F),Mat(rows,cols,CV_64F) };
//			split(EO[o][s], EOPlanes);
//
//
//			waveLength = waveLength * mult;
//		}
//
//
//	}
//	GaborConvolveResult result;
//	result.BP = BP;
//	result.EO = EO;
//	return result;
//}
//
//int  main()
//{
//	Mat img = imread("E:\\house.jpg", IMREAD_GRAYSCALE);
//	//imshow("1",img);
//	cv::Size2d p(0,0);
//	p.height = img.cols;
//	p.width = img.rows;
//	Mat img1=lowpassFilter(img,p, 0.3, 2);
//	imshow("f", img1);
//	waitKey(0);
//	//garborConvolve(img, 3, 4, 12, 2, 30, 1);
//
//}