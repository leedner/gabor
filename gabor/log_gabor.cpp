//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <cmath>
//#include<opencv2/core.hpp>
//
//using  namespace  cv;
//using  namespace  std;
//
//
//// fft�任�����Ƶ�װ���
//void fftshift(cv::Mat& plane0, cv::Mat& plane1)
//{
//	// ���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
//	int cx = plane0.cols / 2;
//	int cy = plane0.rows / 2;
//	cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // Ԫ�������ʾΪ(cx, cy)
//	cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));
//
//	cv::Mat temp;
//	part1_r.copyTo(temp);  //���������½���λ��(ʵ��)
//	part4_r.copyTo(part1_r);
//	temp.copyTo(part4_r);
//
//	part2_r.copyTo(temp);  //���������½���λ��(ʵ��)
//	part3_r.copyTo(part2_r);
//	temp.copyTo(part3_r);
//
//	cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //Ԫ������(cx,cy)
//	cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));
//
//	part1_i.copyTo(temp);  //���������½���λ��(�鲿)
//	part4_i.copyTo(part1_i);
//	temp.copyTo(part4_i);
//
//	part2_i.copyTo(temp);  //���������½���λ��(�鲿)
//	part3_i.copyTo(part2_i);
//	temp.copyTo(part3_i);
//}
//
//cv::Mat ideal_low_kernel(cv::Mat& scr,float cutOff,int n)
//{
//	cv::Mat ideal_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//����,����pow����Ϊfloat�ͣ���Բ�ĵľ��롣
//			ideal_low_pass.at<float>(i, j) = 1 / (1 + pow((d / cutOff), 2 * n));
//		}
//	}
//	ideal_low_pass.at<float>(0, 0) = 1;
//	return ideal_low_pass;
//}
//
//cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur)
//{
//	//cv::Mat mask = scr == scr;
//	//scr.setTo(0.0f, ~mask);
//	//����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
//	cv::Mat plane[] = { Mat_<float>(scr), cv::Mat::zeros(scr.size() , CV_32F)};
//	cv::Mat complexIm;
//	cv::merge(plane, 2, complexIm); // �ϲ�ͨ�� ������������ϲ�Ϊһ��2ͨ����Mat��������
//	int  cn = complexIm.channels();
//	cv::dft(complexIm, complexIm); // ���и���Ҷ�任���������������
//	// ����ͨ����������룩
//	cv::split(complexIm, plane);
//	// ���µĲ�����Ƶ��Ǩ��
//	fftshift(plane[0], plane[1]);
//	
//
//	// *****************�˲���������DFT����ĳ˻�****************
//	cv::Mat blur_r, blur_i, BLUR;
//	cv::multiply(plane[0], blur, blur_r);  // �˲���ʵ�����˲���ģ���ӦԪ����ˣ�
//	cv::multiply(plane[1], blur, blur_i);  // �˲����鲿���˲���ģ���ӦԪ����ˣ�
//	cv::Mat plane1[] = { blur_r, blur_i };
//
//	// �ٴΰ��ƻ���������任
//	fftshift(plane1[0], plane1[1]);
//	cv::merge(plane1, 2, BLUR); // ʵ�����鲿�ϲ�
//	cv::idft(BLUR, BLUR);       // idft���ҲΪ����
//	BLUR = BLUR / BLUR.rows / BLUR.cols;
//	cv::split(BLUR, plane);//����ͨ������Ҫ��ȡͨ��
//	return plane[0];
//}
//
//// ͼ��߽紦��
//cv::Mat image_make_border(cv::Mat& src)
//{
//	int w = cv::getOptimalDFTSize(src.cols); // ��ȡDFT�任����ѿ��
//	int h = cv::getOptimalDFTSize(src.rows); // ��ȡDFT�任����Ѹ߶�
//
//	cv::Mat padded;
//	// ����������ͼ��߽磬���� = 0
//	cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//	padded.convertTo(padded, CV_32FC1);
//	return padded;
//}
//
//
//
//cv::Mat ideal_low_pass_filter(cv::Mat& src)
//{
//	cv::Mat padded = image_make_border(src);
//	cv::Mat ideal_kernel = ideal_low_kernel(padded,0.3,4);
//	cv::Mat result = frequency_filter(padded, ideal_kernel);
//	return result;
//}
//
//int  main()
//{
//	Mat PIC = imread("E:\\house.jpg",0);
//	Mat result = ideal_low_pass_filter(PIC);
//	imshow("origin", PIC);
//	imshow("result", result);
//	waitKey(0);
//}
