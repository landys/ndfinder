#include "siftfeat.h"
#include "imgfeatures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <highgui.h>
#include <cxcore.h>
#include <cv.h>
#include <numeric>
#include <functional>
#include <queue>
#include <map>
#include <cmath>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
using namespace std;
namespace po = boost::program_options; 

string imgName = "E:\\pic_skindetect\\clothtest\\testsift\\19032007144314hello_kitty_tattoo.jpg";
string imgName2 = "E:\\pic_skindetect\\clothtest\\testsift\\hello-kitty.jpg";
string imgMergeName = "E:\\pic_skindetect\\clothtest\\testsift\\merge.jpg";
string outFile = "E:\\pic_skindetect\\clothtest\\testsift\\out.txt";
string outFile2 = "E:\\pic_skindetect\\clothtest\\testsift\\out2.txt";
const int interval = 50;
const int dim = FEATURE_MAX_D;
string outMatchLog = "E:\\pic_skindetect\\clothtest\\testsift\\matchlog.txt";
string outSelectMatchLog = "E:\\pic_skindetect\\clothtest\\testsift\\selectmatchlog.txt";
double threshod = 0.8;
double distlimit = 10.0;
int hideMerged = 0;

int doubleImg;
double contrThr;
double curThr;
double contrWeight;
int maxNkps;

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")

template<class _Ty>
struct minuspower
	: public binary_function<_Ty, _Ty, _Ty>
{	// functor for operator (a-b)(a-b)
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const
	{	// apply operator+ to operands
		return (_Left - _Right) * (_Left - _Right);
	}
};

void doNdbcTest(int argc, char** argv)
{
	if (argc < 6) 
	{
		printf("not enough parameters.");
		return;
	}
	int img_dbl = 0;
	double contr_thr = 0.0;
	int n_max = 0;
	sscanf(argv[3], "%d", &img_dbl);
	sscanf(argv[4], "%lf", &contr_thr);
	sscanf(argv[5], "%d", &n_max);

	showSift(argv[1], argv[2], img_dbl, contr_thr, n_max);
}

IplImage* mergeImages(const IplImage* img1, const IplImage* img2)
{
	CvSize size1 = cvGetSize(img1);
	CvSize size2 = cvGetSize(img2);
	CvSize mergeSize = cvSize(size1.width + size2.width + interval, max(size1.height, size2.height));

	IplImage* imgMerge = cvCreateImage(mergeSize, img1->depth, img1->nChannels);
	cvSet(imgMerge, cvScalar(255, 255, 255));

	cvSetImageROI(imgMerge, cvRect(0, 0, size1.width, size1.height));
	cvCopyImage(img1, imgMerge);
	cvSetImageROI(imgMerge, cvRect(size1.width + interval, 0, size2.width, size2.height));
	cvCopyImage(img2, imgMerge);
	cvResetImageROI(imgMerge);

	return imgMerge;
}

void testMergeImages()
{
	cvNamedWindow("img1");
	cvNamedWindow("img2");
	cvNamedWindow("merged");

	IplImage* img1 = cvLoadImage(imgName.c_str());
	IplImage* img2 = cvLoadImage(imgName2.c_str());

	IplImage* imgMerge = mergeImages(img1, img2);

	cvShowImage("img1", img1);
	cvShowImage("img2", img2);
	cvShowImage("merged", imgMerge);


	cvWaitKey();

	cvDestroyAllWindows();
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&imgMerge);
}

void testSift()
{
	struct feature* fp = 0;
	int n = siftFeature(imgName.c_str(), &fp, doubleImg, contrThr, maxNkps, curThr, contrWeight);

	FILE* fo = fopen(outFile.c_str(), "w");
	for (int i=0; i<n; ++i)
	{
		fprintf(fo, "%d %lg %lg %lg %lg\n", i, fp[i].scl, fp[i].x, fp[i].y, fp[i].ori);
	}
	fclose(fo);

	cvNamedWindow("orig");
	cvNamedWindow("sift");

	IplImage* img = cvLoadImage(imgName.c_str());
	

	IplImage* siftImg = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	cvCopyImage(img, siftImg);

	draw_features(img, fp, n);

	cvShowImage("orig", img);
	cvShowImage("sift", siftImg);

	cvWaitKey();

	cvDestroyAllWindows();
	cvReleaseImage(&img);
	free(fp);
}

double calcDistance(feature* fp1, feature* fp2)
{
	return sqrt((double)inner_product(fp1->descr, fp1->descr+dim, fp2->descr, 0, plus<double>(), minuspower<double>()));
}

void printMatchPair(FILE* fp, const feature& f1, const feature& f21, const feature& f22, double dist1, double dist2, int i)
{
	fprintf(fp, "1st = %lg: %d %lg %lg %lg %lg <==> %lg %lg %lg %lg\n", 
		dist1, i,f1.scl, f1.x, f1.y, f1.ori, 
		f21.scl, f21.x, f21.y, f21.ori);

	fprintf(fp, "2nd = %lg: %d %lg %lg %lg %lg <==> %lg %lg %lg %lg\n", 
		dist2, i,f1.scl, f1.x, f1.y, f1.ori, 
		f22.scl, f22.x, f22.y, f22.ori);
}

void printRatio(FILE* fp, double ratio)
{
	fprintf(fp, "1st/2nd = %lg\n", ratio);
}

void drawMatchLines(IplImage* img, const CvSize& size1, feature* fp1, feature* fp2, int n1, int n2)
{
	FILE* fp = fopen(outMatchLog.c_str(), "w");
	FILE* fpSel = fopen(outSelectMatchLog.c_str(), "w");

	CvScalar c1 = cvScalar(255, 0, 0);
	CvScalar c2 = cvScalar(0, 0, 255);
	for (int i=0; i<n1; ++i)
	{
		map<double, feature*> mt;
		for (int j=0; j<n2; ++j)
		{
			mt[calcDistance(&fp1[i], &fp2[j])] =  &fp2[j];
		}

		map<double, feature*>::iterator it = mt.begin();
		map<double, feature*>::iterator it2 = it;
		it2++;
		printMatchPair(fp, fp1[i], *(it->second), *(it2->second), it->first, it2->first, i);

		double ratio = (it2->first == 0.0 ? 0.0 : it->first/it2->first);
		printRatio(fp, ratio);

		if (ratio <= threshod)
		{
			cvLine(img, cvPoint(fp1[i].x, fp1[i].y), cvPoint(it->second->x+size1.width+interval, it->second->y), c1);
			printMatchPair(fpSel, fp1[i], *(it->second), *(it2->second), it->first, it2->first, i);
			printRatio(fpSel, ratio);
		}
		else if (it->first <= distlimit)
		{
			cvLine(img, cvPoint(fp1[i].x, fp1[i].y), cvPoint(it->second->x+size1.width+interval, it->second->y), c2);
			printMatchPair(fpSel, fp1[i], *(it->second), *(it2->second), it->first, it2->first, i);
			printRatio(fpSel, ratio);
		}
	}

	fclose(fp);
	fclose(fpSel);
}

void showSiftResult(IplImage* img1, IplImage* img2, feature* fp1, feature* fp2, int n1, int n2)
{
	//cvNamedWindow("sift");
	//cvNamedWindow("sift2");
	if (!hideMerged)
	{
		cvNamedWindow("siftMerge");
	}
	

	CvSize size1 = cvGetSize(img1);
	CvSize size2 = cvGetSize(img2);

	IplImage* siftImg1 = cvCreateImage(size1, img1->depth, img1->nChannels);
	cvCopyImage(img1, siftImg1);

	IplImage* siftImg2 = cvCreateImage(size2, img2->depth, img2->nChannels);
	cvCopyImage(img2, siftImg2);

	draw_features(siftImg1, fp1, n1);
	draw_features(siftImg2, fp2, n2);

	IplImage* mergeImg = mergeImages(siftImg1, siftImg2);

	// draw matched lines
	drawMatchLines(mergeImg, size1, fp1, fp2, n1, n2);

	if (!hideMerged)
	{
		cvShowImage("siftMerge", mergeImg);
	}
	cvSaveImage(imgMergeName.c_str(), mergeImg);

	if (!hideMerged)
	{
		cvWaitKey();
		cvDestroyAllWindows();
	}
	cvReleaseImage(&siftImg1);
	cvReleaseImage(&siftImg2);
	cvReleaseImage(&mergeImg);
}

void testSiftMatch()
{
	struct feature* fp = 0;
	int n = siftFeature(imgName.c_str(), &fp, doubleImg, contrThr, maxNkps, curThr, contrWeight);
	struct feature* fp2 = 0;
	int n2 = siftFeature(imgName2.c_str(), &fp2, doubleImg, contrThr, maxNkps, curThr, contrWeight);

	// save feature to the file
	FILE* fo = fopen(outFile.c_str(), "w");
	for (int i=0; i<n; ++i)
	{
		fprintf(fo, "%d %lg %lg %lg %lg\n", i, fp[i].scl, fp[i].x, fp[i].y, fp[i].ori);
	}
	fclose(fo);
	FILE* fo2 = fopen(outFile2.c_str(), "w");
	for (int i=0; i<n2; ++i)
	{
		fprintf(fo2, "%d %lg %lg %lg %lg\n", i, fp2[i].scl, fp2[i].x, fp2[i].y, fp2[i].ori);
	}
	fclose(fo2);

	IplImage* img = cvLoadImage(imgName.c_str());
	IplImage* img2 = cvLoadImage(imgName2.c_str());

	// show result
	showSiftResult(img, img2, fp, fp2, n, n2);
	
	cvReleaseImage(&img);
	cvReleaseImage(&img2);
	free(fp);
	free(fp2);
}

void testSiftMatch(int argc, char** argv)
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("img1,i", po::value<string>(&imgName), "image file 1.")
		("img2,j", po::value<string>(&imgName2), "image file 2.")
		("imgmerge,t", po::value<string>(&imgMergeName), "merged image file name.")
		("double,b", po::value<int>(&doubleImg)->default_value(1), "Double image before sift.")
		("contr,c", po::value<double>(&contrThr)->default_value(0.03), "low contract threshold.")
		("rpc,p", po::value<double>(&curThr)->default_value(10), "ratio of principal curvatures.")
		("contrw,w", po::value<double>(&contrWeight)->default_value(1), "weight of contract, should be in [0,1].")
		("max,m", po::value<int>(&maxNkps)->default_value(3000), "max keypoints per image.")
		("ratiothr,r", po::value<double>(&threshod)->default_value(0.8), "high limit of ratio of distance of closest keypoint and second closest keypoint.")
		("distthr,d", po::value<double>(&distlimit)->default_value(10), "high absolute limit of distance between two keypoints.")
		("hidden,g", po::value<int>(&hideMerged)->default_value(0), "if hide window in the processing, default is not hidden.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("img1") == 0 || vm.count("img2") == 0 
		|| vm.count("imgmerge") == 0)
	{
		cout << desc;
		return;
	}

	outFile = imgName + "_out.txt";
	outFile2 = imgName2 + "_out.txt";

	outMatchLog = imgMergeName + "_match.txt";
	outSelectMatchLog = imgMergeName + "_match_impor.txt";

	testSiftMatch();
}

#ifdef MERGE_TEST
int main(int argc, char** argv)
{
	//doNdbcTest(argc, argv);
	//testSift();
	//testMergeImages();

	testSiftMatch(argc, argv);

	return 0;
}
#endif //MERGE_TEST