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
using namespace std;
namespace po = boost::program_options;

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")

class KeyPoint;


class KeyPoint
{
public:
	int fid;
	float x;
	float y;
	//float scl;
	//float ori;
	vector<int> segs;
};

// used for pruning segment regions.
class SegInfo
{
public:
	int x1; // most left x
	int y1; // most top y
	int x2; // most right x
	int y2; // most bottom y
	int pixels; // number of pixels

	SegInfo() : x1(numeric_limits<int>::max()), y1(numeric_limits<int>::max()), x2(0), y2(0), pixels(0)
	{

	}
};

string InfoFile;
string ImgsFile;
string ResultFile;
string LogFile;
string ImgDir;
string SegDir;

int SegLowLimit;
// for segment region pruning
float Alpha;
float Beta;
float Gama;

ofstream Log;
ofstream Result;

// all keypoints in dataset, index is keypoint id, value is file id.
deque<KeyPoint> KeyPoints;
// key - file id, value - file name
map<int, string> Fnames;
// key - file id, value - keypoint ids
map<int, deque<int> > Fkps;

const int BUF_SIZE=256;
const int BIG_BUF_SIZE=10240;
char buf[BUF_SIZE];
char bigBuf[BIG_BUF_SIZE];


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

string getPathNoExt(const string& fn)
{
	int i = fn.rfind('.');

	return fn.substr(0, i);
}

string Sexts[4] = {"_g2.bmp", "_g3.bmp", "_g4.bmp", "_g5.bmp"};
void getFhSegFiles(const string& imgFile, const string& imgDir, const string& segDir, vector<string>& segFiles)
{
	string s = imgFile;
	s.replace(0, imgDir.length(), segDir);

	string sp = getPathNoExt(s);

	for (int i=0; i<4; ++i)
	{
		segFiles.push_back(sp + Sexts[i]);
	}
}

void initQuery()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	if (fimg == 0)
	{
		printf("Error open %s.\n", ImgsFile.c_str());
		return;
	}
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		Fnames[fid] = string(buf);
	}
	fclose(fimg);
	cout << "load image list finished." << endl;

	FILE* fids = fopen(InfoFile.c_str(), "r");
	if (fids == 0)
	{
		printf("Error open %s.\n", InfoFile.c_str());
		return;
	}
	KeyPoint kp;
	int id;
	int count = 0;
	while (fscanf(fids, "%d %d %f %f", &id, &kp.fid, &kp.x, &kp.y) != EOF)
	{
		KeyPoints.push_back(kp);
		Fkps[kp.fid].push_back(count++);
		// NOTICE if one line of the id file is more than 1024, error will happen.
		fgets(bigBuf, BIG_BUF_SIZE, fids);
	}

	fclose(fids);
	cout << "load file info finished." << endl;
}

int getRgbValue(const IplImage* img, int row, int col)
{
	int offset = col * 3;
	unsigned char b = ((unsigned char*)(img->imageData + img->widthStep * row))[offset];
	unsigned char g = ((unsigned char*)(img->imageData + img->widthStep * row))[offset+1];
	unsigned char r = ((unsigned char*)(img->imageData + img->widthStep * row))[offset+2];

	return r * 65536 + g * 256 + b;
}

void saveSegKeypointsInfo()
{
	FILE* fre = fopen(ResultFile.c_str(), "w");
	if (fre == 0)
	{
		printf("Cannot open %s for write.\n", ResultFile.c_str());
		return ;
	}
	for (int i=0; i<KeyPoints.size(); ++i)
	{
		fprintf(fre, "%d %d %d", i, KeyPoints[i].fid, KeyPoints[i].segs.size());
		for (int j=0; j<KeyPoints[i].segs.size(); ++j)
		{
			fprintf(fre, " %d", KeyPoints[i].segs[j]);
		}
		fprintf(fre, "\n");
	}

	fclose(fre);
}

void segsift()
{
	for (map<int,string>::const_iterator it=Fnames.begin(); it!=Fnames.end(); ++it)
	{
		printf("Process %s.\n", it->second.c_str());

		vector<string> segFiles;
		getFhSegFiles(it->second, ImgDir, SegDir, segFiles);

		deque<vector<int> > segsall;
		for (int i=0; i<segFiles.size(); ++i)
		{
			// key - color of seg, value - keypoint indexes 
			map<int, vector<int> > segs;

			IplImage* img = cvLoadImage(segFiles[i].c_str());
			if (img == 0)
			{
				printf("Error open %s\n", segFiles[i].c_str());
				continue;
			}
			// do pruning
			map<int, SegInfo> segInfos;
			for(int h = 0; h < img->height; ++h) 
			{
				for (int w = 0; w < img->width; ++w)
				{
					int color = getRgbValue(img, h, w);
					SegInfo& segInfo = segInfos[color];
					if (segInfo.x1 > w) segInfo.x1 = w;
					if (segInfo.x2 < w) segInfo.x2 = w;
					if (segInfo.y1 > h) segInfo.y1 = h;
					if (segInfo.y2 < h) segInfo.y2 = h;
					++segInfo.pixels;
				}
			}
			set<int> ignoreSegs;
			int th = img->height * Gama;
			int tw = img->width * Beta;
			int ta = img->height * img->width * Alpha;
			printf("ignore sets: ");
			for (map<int, SegInfo>::const_iterator it2=segInfos.begin(); it2!=segInfos.end(); ++it2)
			{
				if (it2->second.x2 - it2->second.x1 > tw || it2->second.y2 - it2->second.y1 > th
					|| it2->second.pixels > ta)
				{
					ignoreSegs.insert(it2->first);
					printf("%x,", it2->first);
				}
			}
			printf("\n");

			// do
			deque<int>& kps = Fkps[it->first];
			for (int j=0; j<kps.size(); ++j)
			{
				KeyPoint& kp = KeyPoints[kps[j]];
				int color = getRgbValue(img, int(kp.y), int(kp.x));
				if (ignoreSegs.find(color) == ignoreSegs.end())
				{
					segs[color].push_back(kps[j]);
				}
			}
			cvReleaseImage(&img);

			for (map<int, vector<int> >::const_iterator it2=segs.begin(); it2!=segs.end(); ++it2)
			{
				if (it2->second.size() >= SegLowLimit)
				{
					segsall.push_back(vector<int>());
					int si = segsall.size()-1;
					copy(it2->second.begin(), it2->second.end(), back_inserter(segsall[si]));

					// do log
					Log << boost::format("%d %d %x %d ") % si % it->first % it2->first % it2->second.size();
					ostream_iterator<int> lout(Log, ",");
					copy(it2->second.begin(), it2->second.end(), lout);
					Log << endl;
				}
			}	
		}

		for (int i=0; i<segsall.size(); ++i)
		{
			for (int j=0; j<segsall[i].size(); ++j)
			{
				KeyPoints[segsall[i][j]].segs.push_back(i);
			}
		}
	}

	saveSegKeypointsInfo();
}

int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("datamap,m", po::value<string>(&InfoFile), "the map file between pictures and their keypoints.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("imgdir,d", po::value<string>(&ImgDir), "the directory of original images, which will be replaced by SegDir.")
		("segdir,s", po::value<string>(&SegDir), "the directory of segmentation images.")
		("resultFile,r", po::value<string>(&ResultFile), "result file.")
		("logfile,l", po::value<string>(&LogFile), "log file.")
		("lowthreshod,t", po::value<int>(&SegLowLimit)->default_value(3), "keypoints low threshlold in a segmentation region.")
		("alpha,p", po::value<float>(&Alpha)->default_value(0.3), "ratio of area of segment region of image area to be pruned.")
		("beta,b", po::value<float>(&Beta)->default_value(0.6), "ratio of width of segment region of image width to be pruned.")
		("gama,g", po::value<float>(&Gama)->default_value(0.6), "ratio of height of segment region of image height to be pruned.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0 || vm.count("imgdir") == 0 
		|| vm.count("segdir") == 0 || vm.count("resultFile") == 0 || vm.count("logfile") == 0)
	{
		cout << desc;
		return 1;
	}

	Log.open(LogFile.c_str());
	Log << "segid fileid color number keypoints" << endl;

	initQuery();

	printf("begin seg sift points...\n");
	segsift();

	Log.close();
	return 0;
}
