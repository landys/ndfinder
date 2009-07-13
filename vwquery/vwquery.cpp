#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <queue>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <numeric>
#include <utility>
#include <functional>
#include "../sift/siftfeat.h"
#include "../sift/imgfeatures.h"
using namespace std;
namespace po = boost::program_options;

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")
#pragma comment(lib, "sift.lib")

class Node;
class KeyPoint;

// represent non-leaf nodes.
class Node
{
public:
	// if its sons are non-leaf nodes: only minimum and maximum index in the NonLeaves, so the size of sons is 2
	// else: offset order(index) of data file for leaf node
	vector<int> sons;
	// the center of the cluster
	float* center;
	// number of keypoints in the cluster. if a non-leaf node, the number is sum of keypoints number of sub-clusters.
	int npoints;

	Node() : center(0), npoints(0) {}
};

class KeyPoint
{
public:
	int fid;
	float x;
	float y;
	float scl;
	float ori;
};


string InfoFile;
string ImgsFile;
string WordsFile;
string ResultDir;
string LogFile;
string QueryImgsFile;

ofstream Log;

// all keypoints in dataset, index is keypoint id, value is file id.
deque<KeyPoint> KeyPoints;
// key - file id, value - file name
map<int, string> Fnames;
// the first element is root
deque<Node> NonLeaves;
// all keypoints
int N;
// nodes number
int NNodes;
// the real nodes number excluding virtual nodes.
int RealNNodes;
// dimension
int Dim;
// the number of nodes to be watched.
int NWatched;
// query image list read from file
deque<string> QueryImgs;
// partition id
int PartId;
// top k query
int TopK;

int Interval = 20;


int DoubleImg = 1;
double ContrThr = 0.03;
int MaxNkps = 2500;
double CurThr = 10;
double ContrWeight = 0.6;

char buf[256];
char bigBuf[10240];


string& findPicByKpId(int key)
{
	return Fnames[KeyPoints[key].fid];
}

string getFileNameNoExt(const string& fn)
{
	int i = fn.rfind('.');
	int j = fn.rfind('/');
	if (j == -1)
	{
		j = fn.rfind('\\');
	}
	if (i == -1)
	{
		i = fn.length();
	}

	return fn.substr(j+1, i-j-1);
}

string getResultPicName(const string& s1, const string& s2, int n, int index)
{
	return str(boost::format("%s%d_%d_%s_%s.jpg") % ResultDir % index % n % getFileNameNoExt(s1) % getFileNameNoExt(s2));
}

IplImage* mergeImages(const IplImage* img1, const IplImage* img2)
{
	CvSize size1 = cvGetSize(img1);
	CvSize size2 = cvGetSize(img2);
	CvSize mergeSize = cvSize(size1.width + size2.width + Interval, max(size1.height, size2.height));

	IplImage* imgMerge = cvCreateImage(mergeSize, img1->depth, img1->nChannels);
	cvSet(imgMerge, cvScalar(255, 255, 255));

	cvSetImageROI(imgMerge, cvRect(0, 0, size1.width, size1.height));
	cvCopyImage(img1, imgMerge);
	cvSetImageROI(imgMerge, cvRect(size1.width + Interval, 0, size2.width, size2.height));
	cvCopyImage(img2, imgMerge);
	cvResetImageROI(imgMerge);

	return imgMerge;
}

void drawMatchLines(IplImage* img, const CvSize& size1, const deque<pair<int, int> >& mks, feature* feat)
{
	CvScalar color = cvScalar(255, 0, 0);

	for (deque<pair<int, int> >::const_iterator it=mks.begin(); it!=mks.end(); ++it)
	{
		cvLine(img, cvPoint(feat[it->first].x, feat[it->first].y), cvPoint(KeyPoints[it->second].x+size1.width+Interval, KeyPoints[it->second].y), color);

		//Log << boost::format("%g: %g %g %g %g <==> %d %d %g %g %g %g") % it->dist % it->kp1.x % it->kp1.y % it->kp1.scl % it->kp1.ori % it->kp2.id % it->kp2.fid % it->kp2.x % it->kp2.y % it->kp2.scl % it->kp2.ori << endl;
	}
}

void draw_features( IplImage* img, const deque<int>& kids)
{
	CvScalar color = CV_RGB( 255, 255, 255 );
	int i;

	if( img-> nChannels > 1 )
	{
		color = FEATURE_LOWE_COLOR;
	}
	for( i = 0; i < kids.size(); i++ )
	{
		//draw_lowe_feature( img, feat + i, color );
		int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
		double scl, ori;
		double scale = 5.0;
		double hscale = 0.75;
		CvPoint start, end, h1, h2;

		/* compute points for an arrow scaled and rotated by feat's scl and ori */
		start_x = cvRound( KeyPoints[kids[i]].x );
		start_y = cvRound( KeyPoints[kids[i]].y );
		scl = KeyPoints[kids[i]].scl;
		ori = KeyPoints[kids[i]].ori;
		len = cvRound( scl * scale );
		hlen = cvRound( scl * hscale );
		blen = len - hlen;
		end_x = cvRound( len *  cos( ori ) ) + start_x;
		end_y = cvRound( len * -sin( ori ) ) + start_y;
		h1_x = cvRound( blen *  cos( ori + CV_PI / 18.0 ) ) + start_x;
		h1_y = cvRound( blen * -sin( ori + CV_PI / 18.0 ) ) + start_y;
		h2_x = cvRound( blen *  cos( ori - CV_PI / 18.0 ) ) + start_x;
		h2_y = cvRound( blen * -sin( ori - CV_PI / 18.0 ) ) + start_y;
		start = cvPoint( start_x, start_y );
		end = cvPoint( end_x, end_y );
		h1 = cvPoint( h1_x, h1_y );
		h2 = cvPoint( h2_x, h2_y );

		cvLine( img, start, end, color, 1, 8, 0 );
		cvLine( img, end, h1, color, 1, 8, 0 );
		cvLine( img, end, h2, color, 1, 8, 0 );
	}

}

void draw_features( IplImage* img, feature* feat, const deque<int>& kids)
{
	CvScalar color = CV_RGB( 255, 255, 255 );
	int i;

	if( img-> nChannels > 1 )
	{
		color = FEATURE_LOWE_COLOR;
	}
	for( i = 0; i < kids.size(); i++ )
	{
		//draw_lowe_feature( img, feat + i, color );
		int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
		double scl, ori;
		double scale = 5.0;
		double hscale = 0.75;
		CvPoint start, end, h1, h2;

		/* compute points for an arrow scaled and rotated by feat's scl and ori */
		start_x = cvRound( feat[kids[i]].x );
		start_y = cvRound( feat[kids[i]].y );
		scl = feat[kids[i]].scl;
		ori = feat[kids[i]].ori;
		len = cvRound( scl * scale );
		hlen = cvRound( scl * hscale );
		blen = len - hlen;
		end_x = cvRound( len *  cos( ori ) ) + start_x;
		end_y = cvRound( len * -sin( ori ) ) + start_y;
		h1_x = cvRound( blen *  cos( ori + CV_PI / 18.0 ) ) + start_x;
		h1_y = cvRound( blen * -sin( ori + CV_PI / 18.0 ) ) + start_y;
		h2_x = cvRound( blen *  cos( ori - CV_PI / 18.0 ) ) + start_x;
		h2_y = cvRound( blen * -sin( ori - CV_PI / 18.0 ) ) + start_y;
		start = cvPoint( start_x, start_y );
		end = cvPoint( end_x, end_y );
		h1 = cvPoint( h1_x, h1_y );
		h2 = cvPoint( h2_x, h2_y );

		cvLine( img, start, end, color, 1, 8, 0 );
		cvLine( img, end, h1, color, 1, 8, 0 );
		cvLine( img, end, h2, color, 1, 8, 0 );
	}

}

void showResultPerImage(IplImage* img1, IplImage* img2, feature* feat, const deque<pair<int, int> >& mks, int hide, const string& mergedFile)
{
	if (hide == 0)
	{
		cvNamedWindow("siftMerge");
	}

	CvSize size1 = cvGetSize(img1);
	CvSize size2 = cvGetSize(img2);

	IplImage* mergeImg = mergeImages(img1, img2);

	// draw matched lines
	drawMatchLines(mergeImg, size1, mks, feat);

	if (hide == 0)
	{
		cvShowImage("siftMerge", mergeImg);
	}
	cvSaveImage(mergedFile.c_str(), mergeImg);

	if (hide == 0)
	{
		cvWaitKey();
		cvDestroyAllWindows();
	}

	cvReleaseImage(&mergeImg);
}

void readWordsFile()
{
	FILE* fw = fopen(WordsFile.c_str(), "r");
	if (fw == 0)
	{
		printf("Cannot open %s\n", WordsFile.c_str());
		return;
	}

	fscanf(fw, "%d %d %d %d", &N, &NNodes, &RealNNodes, &Dim);
	int id, np, sons;
	for (int i=0; i<NNodes; ++i)
	{
		NonLeaves.push_back(Node());
		int ni = NonLeaves.size() - 1;
		fscanf(fw, "%d %d %d", &id, &NonLeaves[ni].npoints, &sons);
		NonLeaves[ni].sons.resize(sons, 0);
		for (int j=0; j<sons; ++j)
		{
			fscanf(fw, "%d", &NonLeaves[ni].sons[j]);
		}
		NonLeaves[ni].center = new float[Dim];
		for (int j=0; j<Dim; ++j)
		{
			fscanf(fw, "%f", &NonLeaves[ni].center[j]);
		}
	}

	fclose(fw);

	cout << "load words file finished." << endl;
}

void initQuery()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		Fnames[fid] = string(buf);
	}
	fclose(fimg);
	cout << "load image list finished." << endl;

	FILE* fids = fopen(InfoFile.c_str(), "r");
	KeyPoint kp;
	int id;
	while (fscanf(fids, "%d %d %f %f %f %f", &id, &kp.fid, &kp.x, &kp.y, &kp.scl, &kp.ori) != EOF)
	{
		KeyPoints.push_back(kp);
		// NOTICE if one line of the id file is more than 1024, error will happen.
		fgets(bigBuf, 10240, fids);
	}

	fclose(fids);
	cout << "load file info finished." << endl;

	readWordsFile();

	// read query image file
	FILE* fq = fopen(QueryImgsFile.c_str(), "r");
	while (fscanf(fimg, "%s", buf) != EOF)
	{
		QueryImgs.push_back(string(buf));
	}
	fclose(fq);
	cout << "load query image list finished." << endl;
}


void releaseMemory()
{
	for (int i=0; i<NonLeaves.size(); ++i)
	{
		delete[] NonLeaves[i].center;
	}
}

float calcEud2(double* a, float* b)
{
	float sum = 0.0f;
	for (int i=0; i<Dim; ++i)
	{
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sum;
}

// return the index of the real cluster
int queryKeypoint(double* desc)
{
	int begLeaf = NNodes - RealNNodes;
	int curN = 0;
	if (PartId == 1)
	{
		curN = 1;
	}
	else
	{
		curN = 2;
	}
	
	while (curN < begLeaf)
	{
		int nid = 0;
		float dist = std::numeric_limits<float>::max();
		for (int i=NonLeaves[curN].sons[0]; i<=NonLeaves[curN].sons[1]; ++i)
		{
			float d = calcEud2(desc, NonLeaves[i].center);
			if (d < dist)
			{
				nid = i;
				dist = d;
			}
		}
		curN = nid;
	}

	return curN;
}

void copyFile(const string& src, const string& des)
{
	FILE* fs = fopen(src.c_str(), "rb");
	FILE* fd = fopen(des.c_str(), "wb");
	while (true)
	{
		int w = fread(bigBuf, sizeof(char), 10240, fs);
		if (w > 0)
		{
			fwrite(bigBuf, sizeof(char), 10240, fd);
		}
		if (w < 10240)
		{
			break;
		}
	}
	fclose(fs);
	fclose(fd);
}

void queryTest()
{
	feature* feat = 0;
	int n = 0; // size of sift features of test image
	for (int i=0; i<QueryImgs.size(); ++i)
	{
		printf("Query %s...\n", QueryImgs[i].c_str());
		n = siftFeature(QueryImgs[i].c_str(), &feat, DoubleImg, ContrThr, MaxNkps, CurThr, ContrWeight);
		// key - fid, value - matched keypoint id pairs, notice the first id is index of features.
		map<int, deque<pair<int, int> > > result;
		for (int i=0; i<n; ++i)
		{
			int ni = queryKeypoint(feat[i].descr);
			for (int j=0; j<NonLeaves[ni].sons.size(); ++j)
			{
				int id = NonLeaves[ni].sons[j];
				KeyPoint kp = KeyPoints[id];
				if (result.find(kp.fid) == result.end())
				{
					result.insert(pair<int, deque<pair<int, int> >>(kp.fid, deque<pair<int, int> >()));
				}
				result[kp.fid].push_back(pair<int, int>(i, id));
			}
		}

		priority_queue<pair<int, int>, vector<pair<int, int> >, greater<pair<int, int> > > pq;
		for (map<int, deque<pair<int, int> > >::const_iterator it=result.begin(); it!=result.end(); ++it)
		{
			pq.push(pair<int, int>(it->second.size(), it->first));
			if (pq.size() > TopK)
			{
				pq.pop();
			}
		}
		
		while (!pq.empty())
		{
			pair<int, int> p = pq.top();
			deque<pair<int, int> >& kpps = result[p.second];
			string reimgfile = str(boost::format("%s\\%d_%d_%s.jpg") % ResultDir.c_str() % i % p.first % getFileNameNoExt(Fnames[p.second]).c_str());

			deque<int> qids;
			deque<int> kids;
			for (int j=0; j<kpps.size(); ++j)
			{
				qids.push_back(kpps[j].first);
				kids.push_back(kpps[j].second);
			}

			IplImage* img = cvLoadImage(QueryImgs[i].c_str());
			draw_features(img, feat, qids);

			IplImage* img2 = cvLoadImage(Fnames[p.second].c_str());
			draw_features(img2, kids);

			showResultPerImage(img, img2, feat, kpps, 1, reimgfile);


			//copyFile(Fnames[p.second], reimgfile);

			pq.pop();

			cvReleaseImage(&img);
			cvReleaseImage(&img2);
		}

		free(feat);
	}

}


int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("queryimgs,q", po::value<string>(&QueryImgsFile), "the query image list.")
		("datamap,m", po::value<string>(&InfoFile), "the map file between pictures and their keypoints.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("wordfile,w", po::value<string>(&WordsFile), "sift visual words index file.")
		("resultdir,r", po::value<string>(&ResultDir), "result directory.")
		("logfile,l", po::value<string>(&LogFile), "log file.")
		("parts,p", po::value<int>(&PartId)->default_value(1), "partition id, 1 for p1, 2 for p2")
		("topk,k", po::value<int>(&TopK)->default_value(50), "top k results")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("queryimgs") == 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0
		|| vm.count("wordfile") == 0 || vm.count("resultdir") == 0 || vm.count("logfile") == 0)
	{
		cout << desc;
		return 1;
	}

	Log.open(LogFile.c_str(), ios_base::app);

	initQuery();

	printf("begin query test...\n");
	queryTest();

	releaseMemory();

	Log.close();
	return 0;
}
