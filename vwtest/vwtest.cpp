#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
using namespace std;
namespace po = boost::program_options;

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")

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
	//float* center;
	// number of keypoints in the cluster. if a non-leaf node, the number is sum of keypoints number of sub-clusters.
	//int npoints;

	//Node() : center(0), npoints(0) {}
};

class KeyPoint
{
public:
	int fid;
	float x;
	float y;
	//float scl;
	//float ori;
};

string InfoFile;
string ImgsFile;
string WordsFile;
string ResultDir;
string LogFile;

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

char bigBuf[10240];

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
		//fscanf(fw, "%d %d %d", &id, &NonLeaves[ni].npoints, &sons);
		fscanf(fw, "%d %d %d", &id, &np, &sons);
		NonLeaves[ni].sons.resize(sons, 0);
		for (int j=0; j<sons; ++j)
		{
			fscanf(fw, "%d", &NonLeaves[ni].sons[j]);
		}
		fgets(bigBuf, 10240, fw);
		/*NonLeaves[ni].center = new float[Dim];
		for (int j=0; j<Dim; ++j)
		{
			fscanf(fw, "%f", &NonLeaves[ni].center[j]);
		}*/
	}

	fclose(fw);

	cout << "load words file finished." << endl;
}

void initQuery()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	char buf[256];
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
	while (fscanf(fids, "%d %d %f %f", &id, &kp.fid, &kp.x, &kp.y) != EOF)
	{
		KeyPoints.push_back(kp);
		// NOTICE if one line of the id file is more than 1024, error will happen.
		fgets(bigBuf, 10240, fids);
	}

	fclose(fids);
	cout << "load file info finished." << endl;

	readWordsFile();
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

void watchWords()
{
	for (int i=0; i<NWatched; ++i)
	{
		int ni = NNodes - i - 1;
		for (int j=0; j<NonLeaves[ni].sons.size(); ++j)
		{
			int id = NonLeaves[ni].sons[j];
			KeyPoint kp = KeyPoints[id];
			IplImage* img = cvLoadImage(Fnames[kp.fid].c_str());
			// 16*16 area
			int x1 = kp.x > 8 ? kp.x - 8 : 0;
			int x2 = kp.x + 8 < img->width ? kp.x + 8 : img->width - 1;
			int y1 = kp.y > 8 ? kp.y - 8 : 0;
			int y2 = kp.y + 8 < img->height ? kp.y + 8 : img->height - 1;

			CvScalar color = cvScalar(0, 255, 0);
			cvLine(img, cvPoint(x1, y1), cvPoint(x1, y2), color);
			cvLine(img, cvPoint(x2, y2), cvPoint(x1, y2), color);
			cvLine(img, cvPoint(x1, y1), cvPoint(x2, y1), color);
			cvLine(img, cvPoint(x2, y2), cvPoint(x2, y1), color);

			string reimgfile = str(boost::format("%s\\%d_%s.jpg") % ResultDir.c_str() % ni % getFileNameNoExt(Fnames[kp.fid]).c_str());
			cvSaveImage(reimgfile.c_str(), img);
			cvReleaseImage(&img);
		}
	}
}

void releaseMemory()
{
	/*for (int i=0; i<NonLeaves.size(); ++i)
	{
		delete[] NonLeaves[i].center;
	}*/
}


int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("datamap,m", po::value<string>(&InfoFile), "the map file between pictures and their keypoints.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("wordfile,w", po::value<string>(&WordsFile), "sift visual words index file.")
		("resultdir,r", po::value<string>(&ResultDir), "result directory.")
		("logfile,l", po::value<string>(&LogFile), "log file.")
		("nodes,n", po::value<int>(&NWatched)->default_value(1), "number of nodes to be watched.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0
		|| vm.count("wordfile") == 0 || vm.count("resultdir") == 0 || vm.count("logfile") == 0)
	{
		cout << desc;
		return 1;
	}

	initQuery();

	printf("begin watch nodes...\n");
	watchWords();

	releaseMemory();

	return 0;
}
