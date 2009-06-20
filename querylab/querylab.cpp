//#define PCASIFT

#ifdef PCASIFT
#include "../pcasift/siftfeat.h"
#include "../pcasift/imgfeatures.h"
#else
#include "../sift/siftfeat.h"
#include "../sift/imgfeatures.h"
#endif // PCASIFT

#include <iostream>
#include <cstdio>
#include <fstream>
#include <limits>
#include <lshkit.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <deque>
#include <map>
#include <set>
#include <utility>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <vector>
using namespace std;
using namespace lshkit;
namespace po = boost::program_options; 

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")

#ifdef PCASIFT
#pragma comment(lib, "pcasift.lib")
#else
#pragma comment(lib, "sift.lib")
#endif // PCASIFT

string IndexFile;
string DataFile;
string InfoFile;
string ImgsFile;
string ResultFile;
string QueryLog;

string PicDir;
string TestPics;


#ifdef PCASIFT
string EigsFile;
const float DefaultDistLimit = 0.5f;
const int DIM = PCASIZE;
#else
const float DefaultDistLimit = 100;
const int DIM = 128;
#endif // PCASIFT

const float R = 1e30;//numeric_limits<float>::max();
const unsigned T = 20;
int K = 60;
int TopKPics = 200;
//const int Interval = 50;
int DownMatchKpLimit = 2;
int UpMatchKpLimit = 10; // another match limit
float DistLimit = DefaultDistLimit;

int DoubleImg = 1;
double ContrThr = 0.03;
int MaxNkps = 100;
double CurThr = 10;
double ContrWeight = 1.0;

// a keypoint in file with id "fid" is matched with the distance "dist".
class MatchKp
{
public:
	int fid;
	float dist;

	MatchKp()
	{
		fid = 0;
		dist = 1e30;// numeric_limits<float>::max();
	}

	MatchKp(int fid,  float dist) : fid(fid), dist(dist)
	{

	}
};

bool operator<(const MatchKp& mk1, const MatchKp& mk2)
{
	return mk1.dist < mk2.dist;
}


// all keypoints in dataset, index is keypoint id, value is file id.
deque<int> KeyPoints;
// key - file id, value - file name
map<int, string> Fnames;
// key - file id, value - if the most similar key is found in this file for a certain keypoint.
//map<int, bool> FFound;
// key - file id, value - keypoint id, which is the index of KeyPoints
//map<int, deque<int> > Fkps;

// for draw merged file

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
	char bigBuf[10240];
	int id;
	while (fscanf(fids, "%d %d", &id, &fid) != EOF)
	{
		KeyPoints.push_back(fid);
		// NOTICE if one line of the id file is more than 10240, error will happen.
		fgets(bigBuf, 10240, fids);
	}

	fclose(fids);
	cout << "load file info finished." << endl;
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

bool checkMatchByFileName(const string& q, const string& r)
{
	string rr = getFileNameNoExt(r);
	return (rr.length() > q.length() && rr[q.length()] == '_' && rr.find(q) == 0);
}

void queryImages(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("querylist,q", po::value<string>(&TestPics), "the query picture list file.")
#ifdef PCASIFT
		("eigs,e", po::value<string>(&EigsFile), "the eigespace file to initialize PCA-SIFT.")
#endif // PCASIFT
		("index,i", po::value<string>(&IndexFile), "the mplsh index file.")
		("data,d", po::value<string>(&DataFile), "the data of the index.")
		("datamap,m", po::value<string>(&InfoFile), "the map file between pictures and their keypoints.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("resultfile,r", po::value<string>(&ResultFile), "the test result file.")
		("log,l", po::value<string>(&QueryLog), "the query result log.")
		("maxnkps,n", po::value<int>(&MaxNkps)->default_value(100), "max number of keypoints per image when sift.")
		("double,b", po::value<int>(&DoubleImg)->default_value(1), "Double image before sift.")
		("contr,c", po::value<double>(&ContrThr)->default_value(0.03), "low contract threshold.")
		("rpc,p", po::value<double>(&CurThr)->default_value(10), "ratio of principal curvatures.")
		("contrw,w", po::value<double>(&ContrWeight)->default_value(1), "weight of contract, should be in [0,1].")
		("topk,k", po::value<int>(&K)->default_value(60), "the top k points by every keypoint query.")
		("distlimit,t", po::value<float>(&DistLimit)->default_value(DefaultDistLimit), "distance limit between keypoints.")
		("downmatchlimit,g", po::value<int>(&DownMatchKpLimit)->default_value(2), "the down number of keypoints should match between near-duplicate images. it uses all matchlimit in [downmatchlimit,upmatchlimt] to calculate all precision/recall.")
		("upmatchlimit,s", po::value<int>(&UpMatchKpLimit)->default_value(10), "the down number of keypoints should match between near-duplicate images.");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0  || vm.count("querylist") == 0 || vm.count("index") == 0 
		|| vm.count("data") == 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0 || vm.count("resultfile") == 0 || vm.count("log") == 0)
	{
		cout << desc;
		return;
	}

	// load data
	cout << "LOADING DATA..." << endl;
	FloatMatrix data(DataFile);
	typedef MultiProbeLshIndex<FloatMatrix::Accessor> Index;

	FloatMatrix::Accessor accessor(data);
	Index index(accessor);

	// load index
	ifstream is(IndexFile.c_str(), ios_base::binary);
	if (is) {
		is.exceptions(ios_base::eofbit | ios_base::failbit | ios_base::badbit);
		cout << "LOADING INDEX..." << endl;
		//timer.restart();
		index.load(is);
		verify(is);
		//cout << boost::format("LOAD TIME: %1%s.") % timer.elapsed() << endl;
	}

#ifdef PCASIFT
	// init for pca-sift, never forget again!!!!
	initialeigs(EigsFile.c_str());
#endif // PCASIFT

	ofstream qlog;
	qlog.open(QueryLog.c_str(), ios_base::app);
	qlog << "/******************************************************/" << endl;

	ofstream outRe(ResultFile.c_str());
	outRe << "filename,sifttime,querytime";
	for (int i=DownMatchKpLimit; i<=UpMatchKpLimit; ++i)
	{
		outRe << boost::format(",correct%d,wrong%d") % i % i;
	}
	outRe << endl;

	cout << "init query..." << endl;
	initQuery();
	cout << "finish init query." << endl;

	//Stat recall;
	Stat cost;
	Topk<unsigned> topk;
	feature* feat = 0;
	string testPic;
	ifstream qimgs(TestPics.c_str());
	while (getline(qimgs, testPic))
	{
		cout << "Process pic: " << testPic.c_str() << endl;
		qlog << "Process pic: " << testPic.c_str() << endl;
		outRe << testPic.c_str();

		long time = clock();
		int Q = siftFeature(testPic.c_str(), &feat, DoubleImg, ContrThr, MaxNkps, CurThr, ContrWeight);
		outRe << "," << clock()-time;

		cout << "SIFT finish, begin query..." << endl;

		// key - pic file id, value - matched keypoits sored by distance. multiset allows the same distance of different matched keypoints.
		map<int, multiset<MatchKp> > results;

		//boost::progress_display progress(Q);
		time = clock();
		float qd[DIM];
		for (int i = 0; i < Q; ++i)
		{
			unsigned cnt;
			topk.reset(K, R);
			for (int j=0; j<DIM; ++j)
			{
	#ifdef PCASIFT
				qd[j] = feat[i].PCAdescr[j];
	#else
				qd[j] = feat[i].descr[j];
	#endif // PCASIFT
			}
			index.query(qd, &topk, T, &cnt);

			// show result
			for (int j=0; j<topk.size(); ++j)
			{
				if (topk[j].key >= KeyPoints.size())
				{
					qlog << boost::format("wrong match: (%g, %g, %g, %g) == (%d, %g)") % feat[i].x % feat[i].y % feat[i].scl % feat[i].ori % topk[j].key % topk[j].dist << endl;
					break;
				}
				int fid = KeyPoints[topk[j].key];
				// already get the closest keypoint in the file
				if (topk[j].dist > DistLimit)
				{
					continue;
				}
				map<int, multiset<MatchKp> >::const_iterator it = results.find(fid);
				if (it == results.end())
				{
					results[fid] = multiset<MatchKp>();
				}

				results[fid].insert(MatchKp(fid, topk[j].dist));
			}

			//recall << bench.getAnswer(i).recall(topk);

			cost << double(cnt)/double(data.getSize());
			//++progress;
		}

		//qlog << "[RECALL] " << recall.getAvg() << " +/- " << recall.getStd() << endl;
		qlog << "[COST] " << cost.getAvg() << " +/- " << cost.getStd() << endl;
		cout << "[COST] " << cost.getAvg() << " +/- " << cost.getStd() << endl;

		qlog << "Raw matched Results: " << results.size() << endl;
		cout << "Raw matched Results: " << results.size() << endl;

		// calculate results
		multimap<int, int, greater<int> > topkM;
		for (map<int, multiset<MatchKp> >::const_iterator it=results.begin(); it!=results.end(); ++it)
		{
			int mps = it->second.size();
			if (mps >= DownMatchKpLimit)
			{
				topkM.insert(pair<int, int>(mps, it->first));
			}
		}

		// do results statistic
		int count = 0;
		int* corrects = new int[UpMatchKpLimit+1];
		int* wrongs = new int[UpMatchKpLimit+1];
		fill(&corrects[0], &corrects[UpMatchKpLimit+1], 0);
		fill(&wrongs[0], &wrongs[UpMatchKpLimit+1], 0);
		vector<int> correctMatches;
		vector<int> wrongMatches;
		const string testPicPart = getFileNameNoExt(testPic);
		for (multimap<int, int, greater<int> >::const_iterator it=topkM.begin(); it!=topkM.end(); ++it)
		{
			if (count >= TopKPics)
			{
				break;
			}
			++count;

			string& sr = Fnames[it->second];

			if (checkMatchByFileName(testPicPart, sr))
			{
				correctMatches.push_back(it->first);
				for (int i=DownMatchKpLimit; i<=UpMatchKpLimit; ++i)
				{
					if (it->first >= i)
					{
						++corrects[i];
					}
				}
			}
			else
			{
				wrongMatches.push_back(it->first);
				for (int i=DownMatchKpLimit; i<=UpMatchKpLimit; ++i)
				{
					if (it->first >= i)
					{
						++wrongs[i];
					}
				}
			}
			
		}

		cout << "total real match files: " << count << endl;
		qlog << "total real match files: " << count << endl;

		outRe << boost::format(",%ld") % (clock()-time);
		for (int i=DownMatchKpLimit; i<=UpMatchKpLimit; ++i)
		{
			outRe << boost::format(",%d,%d") % corrects[i] % wrongs[i];
		}
		outRe << endl;
		qlog << "correct matched points: " << endl;
		for (int i=0; i<correctMatches.size(); ++i)
		{
			qlog << correctMatches[i] << ",";
		}
		qlog << endl << "wrong matched points: " << endl;
		for (int i=0; i<wrongMatches.size(); ++i)
		{
			qlog << wrongMatches[i] << ",";
		}
		qlog << endl;

		delete[] corrects;
		delete[] wrongs;
		free(feat);
	}

	qimgs.close();
	outRe.close();
	qlog.close();
}


int main(int argc, char* argv[])
{
	queryImages(argc, argv);

	return 0;
}
