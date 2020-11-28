#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include "util.h"

using namespace std;
using namespace cv;

int find(vector<node>& t, uchar threshold)
{
	int l = 0, r = static_cast<int>(t.size()) - 1;
	while (l < r) {
		int mid = (l + r + 1) >> 1;
		if (t[mid].val >= threshold) l = mid;
		else r = mid - 1;
	}
	return r;
}

Vec3b atmosphic(Mat pic, Mat img)
{
	vector<node> t;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) {
			node p = node(i, j, img.at<uchar>(i, j));
			t.push_back(p);
		}
	}
	sort(t.begin(), t.end());
	double p = 0.001;
	int n = static_cast<int>((t.size()) * p);
	uchar threshold = t[n].val;
	Vec3b A = Vec3b(0, 0, 0);
	int index = find(t, threshold);
	vector<Mat> channels;
	split(pic, channels);
	for (size_t i = 0; i < pic.channels(); i++) {
		Mat pic_t = channels.at(i);
		for (int j = 0; j <= index; j++) {
			A[i] = max(A[i], pic_t.at<uchar>(t[j].x, t[j].y));
		}
	}
	//cout << int(A[0]) << " " << int(A[1]) << " " << int(A[2]) << endl;
	return A;
}
