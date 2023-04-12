#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

Mat logGaborFilter(Size size, double sigma, double f, double theta, double sigma_x, double sigma_y)
{
    Mat kernel(size, CV_64F);
    double half_width = size.width / 2.0;
    double half_height = size.height / 2.0;
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    for (int y = 0; y < size.height; y++)
    {
        for (int x = 0; x < size.width; x++)
        {
            double x_prime = (x - half_width) * cos_theta + (y - half_height) * sin_theta;
            double y_prime = -(x - half_width) * sin_theta + (y - half_height) * cos_theta;
            double r = sqrt(pow(x_prime, 2.0) + pow(y_prime, 2.0));
            double value = exp(-pow(log(r / f), 2.0) / (2.0 * pow(log(sigma), 2.0))) * exp(-pow(x_prime, 2.0) / (2.0 * pow(sigma_x, 2.0))) * exp(-pow(y_prime, 2.0) / (2.0 * pow(sigma_y, 2.0)));
            kernel.at<double>(y, x) = value;
        }
    }

    return kernel;
}

int main()
{
    Mat img = imread("path/to/image.jpg", IMREAD_GRAYSCALE);
    Mat kernel = logGaborFilter(img.size(), 1.0, 0.5, 0.0, 1.0, 1.0);
    Mat filtered_img;
    filter2D(img, filtered_img, CV_64F, kernel);

    imshow("Original Image", img);
    imshow("Filtered Image", filtered_img);
    waitKey(0);

    return 0;
}
