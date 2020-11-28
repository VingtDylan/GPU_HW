#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct node {
	int x;
	int y;
	uchar val;
	node() {};
	node(int _x, int _y, int _val) : x(_x), y(_y), val(_val) {};
	bool operator < (const node & b) {
		return val > b.val;
	}
};

int find(vector<node>& t, uchar threshold);

Vec3b atmosphic(Mat pic, Mat img);