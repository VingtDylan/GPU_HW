#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "util.h"

#define _CRT_SECURE_NO_WARNINGS

using namespace std;
using namespace cv;

Mat minChannel(Mat img);
Mat darkChannel(Mat img);
Mat transmission(Mat img, Vec3b a, float w, float t0);
Mat recover(Mat img, Mat guide, Vec3b A);
Mat deHaze(Mat img);

int main1() {
	Mat hazed = imread("extra1.jpg");

	clock_t start_t = clock();
	Mat dehazed = deHaze(hazed);
	clock_t end_t = clock();

	double total_t;
	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
	cout << "CPU 运行占用的总时间为: " << total_t << " ms" << endl;

	imshow("hazed image", hazed);
	imshow("Dehazed image", dehazed);
	waitKey(0);
	return 0;
}

Mat minChannel(Mat img)
{
	CV_Assert(!img.empty());
	//开始计时
	double start_t = clock();
	Mat _minChannel = Mat::zeros(img.size(), CV_8UC1);

	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			_minChannel.at<uchar>(i, j) = img.at<Vec3b>(i, j)[0];
			_minChannel.at<uchar>(i, j) = min(_minChannel.at<uchar>(i, j), img.at<Vec3b>(i, j)[1]);
			_minChannel.at<uchar>(i, j) = min(_minChannel.at<uchar>(i, j), img.at<Vec3b>(i, j)[2]);
		}
	}
	//停止计时
	double end_t = clock();
	cout << "minChannel：" << (end_t - start_t) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
	return _minChannel;
}

Mat darkChannel(Mat img)
{
	CV_Assert(!img.empty());
	//开始计时
	double start_t = clock();
	int r = 7;
	Mat _kernel = getStructuringElement(MORPH_RECT, Size(2 * r + 1, 2 * r + 1));
	int m = (_kernel.rows + 1) / 2;
	int n = (_kernel.cols + 1) / 2;
	Mat _darkChannel(img.size(), img.type());
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			int xl = max(0, i - m + 1);
			int xr = min(i + m - 1, img.rows);
			int yl = max(0, j - n + 1);
			int yr = min(j + n - 1, img.cols);
			uchar t_inf = 0xff;
			for (int k = xl; k < xr; ++k)
			{
				for (int l = yl; l < yr; ++l)
				{
					uchar s = img.at<uchar>(k, l);
					t_inf = t_inf > s ? s : t_inf;
				}
			}
			_darkChannel.at<uchar>(i, j) = t_inf;
		}
	}
	//停止计时
	double end_t = clock();
	cout << "darkChannel：" << (end_t - start_t) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
	return _darkChannel;
}

Mat transmission(Mat img, Vec3b a, float w, float t0)
{
	CV_Assert(!img.empty());
	//开始计时
	double start_t = clock();
	vector<Mat> channels;
	split(img, channels);
	Mat t_hat(channels[0].size(), CV_32F);
	for (int i = 0; i < t_hat.rows; i++)
	{
		for (int j = 0; j < t_hat.cols; j++)
		{
			t_hat.at<float>(i, j) = (float)channels[0].at<uchar>(i, j) / a[0];
			t_hat.at<float>(i, j) = min(t_hat.at<float>(i, j), (float)channels[1].at<uchar>(i, j) / a[1]);
			t_hat.at<float>(i, j) = min(t_hat.at<float>(i, j), (float)channels[2].at<uchar>(i, j) / a[2]);
		}
	}
	for (int i = 0; i < t_hat.rows; i++)
	{
		for (int j = 0; j < t_hat.cols; j++)
		{
			t_hat.at<float>(i, j) = max(t0, 1 - w * t_hat.at<float>(i, j));
		}
	}
	//停止计时
	double end_t = clock();
	cout << "transmission：" << (end_t - start_t) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
	return t_hat;
}

Mat recover(Mat img, Mat guide, Vec3b A)
{
	CV_Assert(!img.empty());
	//开始计时
	double start_t = clock();
	vector<Mat> channels;
	split(img, channels);
	for (int c = 0; c < channels.size(); c++)
	{
		Mat pic_t = channels.at(c);
		for (int i = 0; i < channels[c].rows; i++)
		{
			for (int j = 0; j < channels[c].cols; j++)
			{
				pic_t.at<uchar>(i, j) = uchar(min(0xff, int(uchar((pic_t.at<uchar>(i, j) - A[c]) / guide.at<float>(i, j) + A[c]))));
			}
		}
	}
	Mat _recover;
	merge(channels, _recover);
	//停止计时
	double end_t = clock();
	cout << "recover：" << (end_t - start_t) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
	return _recover;
}

Mat deHaze(Mat img) {
	CV_Assert(img.channels() == 3);

	Mat _minChannel = minChannel(img);
	Mat _darkChannel = darkChannel(_minChannel);
	Vec3b _A = atmosphic(img, _darkChannel);
	float w = 0.85f, t0 = 0.1f;
	Mat _tx = transmission(img, _A, w, t0);
	Mat _recover = recover(img, _tx, _A);
	return _recover;
}
