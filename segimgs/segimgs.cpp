#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <cstdio>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
using namespace std;

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")

const string seg1 = "E:\\testpics\\101_ObjectCategories\\buddha\\image_0058.jpg";

void testPyrSegmentation(IplImage* img)
{
	//cvSetImageROI(img, cvRect(0, 0, 480, 480));

	IplImage* img2 = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);
	//IplImage* img3 = cvCreateImage(cvSize(480, 480), IPL_DEPTH_32S, 1);

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* comp = 0;
	cvPyrSegmentation(img, img2, storage, &comp, 4, 200, 50);

	cout << comp->total << endl;
	//cvWatershed(img, img3);

	cvNamedWindow("img");
	cvNamedWindow("img2");
	//cvNamedWindow("img3");

	cvShowImage("img", img);
	cvShowImage("img2", img2);
	//cvShowImage("img3", img3);

	cvWaitKey();

	cvDestroyAllWindows();
	cvReleaseImage(&img);
	cvReleaseImage(&img2);
	//cvReleaseImage(&img3);
}

void testMeanShift(IplImage* img)
{
	//cvSetImageROI(img, cvRect(0, 0, 480, 480));

	IplImage* img2 = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);
	IplImage* img3 = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* comp = 0;
	cvPyrMeanShiftFiltering(img, img2, 20, 40, 2);
	//cvPyrSegmentation(img2, img3, storage, &comp, 4, 200, 50);
	//cout << comp->total << endl;

	cvNamedWindow("img");
	cvNamedWindow("img2");
	cvNamedWindow("img3");

	cvShowImage("img", img);
	cvShowImage("img2", img2);
	cvShowImage("img3", img3);

	cvWaitKey();

	cvDestroyAllWindows();
	cvReleaseImage(&img);
	cvReleaseImage(&img2);
	cvReleaseImage(&img3);

}

int main(int argc, char* argv[])
{
	IplImage* img = cvLoadImage(seg1.c_str());
	//testPyrSegmentation(img);
	testMeanShift(img);

	return 0;
}
