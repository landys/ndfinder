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
	// if its sons are non-leaf nodes: key - only minimum and maximum index in the NonLeaves, so the size of sons is 2, value - 0.0f.
	// else: key - offset order(index) of data file for leaf node, value - tf weight.
	vector<pair<int, float> > sons;
	// the center of the cluster
	float* center;
	// number of keypoints in the cluster. if a non-leaf node, the number is sum of keypoints number of sub-clusters.
	int npoints;
	// idf of the word
	float idf;

	Node() : center(0), npoints(0), idf(0.0f) {}
};

class KeyPoint
{
public:
	int fid;
	float x;
	float y;
	float scl;
	float ori;
	float tfidf;
	KeyPoint() : fid(0), x(0.0f), y(0.0f), scl(0.0f), ori(0.0f), tfidf(0.0f) {}
};

string InfoFile;
string ImgsFile;
string WordsFile;
string ResultDir;
string ResultFile;
string LogFile;
string QueryImgsFile;

ofstream Log;
ofstream Result;

// all keypoints in dataset, index is keypoint id, value is file id.
deque<KeyPoint> KeyPoints;
// key - file id, value - file name
map<int, string> Fnames;
// key - fild id, value - standard factors
map<int, float> Factors;
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
deque<string> QueryImgs1;
// query image list read from file
deque<string> QueryImgs2;
// partition id
int PartId;
// top k query
int TopK;

class Score
{
public:
	int correct;
	int wrong;
	float sumScore;

	Score() : correct(0), wrong(0), sumScore(0.0f) {}

	void add(int correct, int wrong, float score)
	{
		this->correct += correct;
		this->wrong += wrong;
		this->sumScore += score;
	}

	float score() const
	{
		return (correct + wrong == 0) ? 0 : sumScore * TopK / (correct + wrong);
	}
};
// result scores, string-category type, score-score of category
map<string, Score> Scores;

int Interval = 20;

float Wis[4] = {2.0, 1.5, 1.0, 0.5};
float SumWi = (2.0 + 1.5 + 1.0 + 0.5) * 5;

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

// the parameter fn should has the result. Face and Face_esay is NOT the same category.
string getObjectCategoryFromPath(const string& fn)
{
	int i = fn.rfind('/');
	if (i == -1)
	{
		i = fn.rfind('\\');
	}

	int j = fn.rfind('/', i-1);
	if (j == -1)
	{
		j = fn.rfind('\\', i-1);
	}

	string re = fn.substr(j+1, i-j-1);

	/*	i = re.find('_');
	if (i != -1)
	{
	re = re.substr(0, i);
	}*/

	return re;
}

// the parameter fn should has the result. Face and Face_esay is the same type.
string getObjectTypeFromPath(const string& fn)
{
	int i = fn.rfind('/');
	if (i == -1)
	{
		i = fn.rfind('\\');
	}

	int j = fn.rfind('/', i-1);
	if (j == -1)
	{
		j = fn.rfind('\\', i-1);
	}

	string re = fn.substr(j+1, i-j-1);

	i = re.find('_');
	if (i != -1)
	{
		re = re.substr(0, i);
	}

	return re;
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

// invoked after read InfoFile
void readWordsFile()
{
	FILE* fw = fopen(WordsFile.c_str(), "r");
	if (fw == 0)
	{
		printf("Cannot open %s\n", WordsFile.c_str());
		return;
	}

	fscanf(fw, "%d %d %d %d", &N, &NNodes, &RealNNodes, &Dim);
	int id, np, ns;
	for (int i=0; i<NNodes; ++i)
	{
		NonLeaves.push_back(Node());
		int ni = NonLeaves.size() - 1;
		fscanf(fw, "%d %d %f %d", &id, &NonLeaves[ni].npoints, &NonLeaves[ni].idf, &ns);
		NonLeaves[ni].sons.resize(ns);
		for (int j=0; j<ns; ++j)
		{
			fscanf(fw, "%d %f", &NonLeaves[ni].sons[j].first, &NonLeaves[ni].sons[j].second);
			// get tfidf
			KeyPoints[NonLeaves[ni].sons[j].first].tfidf = NonLeaves[ni].sons[j].second * NonLeaves[ni].idf;
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

void calFactors()
{
	cout << "begin calculate factors of image..." << endl;
	for (int i=0; i<KeyPoints.size(); ++i)
	{
		Factors[KeyPoints[i].fid] += KeyPoints[i].tfidf * KeyPoints[i].tfidf;
	}

	for (map<int, float>::iterator it=Factors.begin(); it!=Factors.end(); ++it)
	{
		it->second = sqrt((double)it->second);
	}

	cout << "calculate factors of images finished: " <<  Factors.size() << endl;
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

	// read tfidf word file
	readWordsFile();

	// calculate standard factor for images
	calFactors();

	// read query image file
	FILE* fq = fopen(QueryImgsFile.c_str(), "r");
	int type;
	while (fscanf(fimg, "%d %s", &type, buf) != EOF)
	{
		if (type == 1)
		{
			QueryImgs1.push_back(string(buf));
		}
		else
		{
			QueryImgs2.push_back(string(buf));
		}
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
int queryKeypoint(double* desc, int curN)
{
	int begLeaf = NNodes - RealNNodes;

	while (curN < begLeaf)
	{
		int nid = 0;
		float dist = std::numeric_limits<float>::max();
		for (int i=NonLeaves[curN].sons[0].first; i<=NonLeaves[curN].sons[1].first; ++i)
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


void queryOneImg(const string& imgfile, int type)
{
	printf("Query %s...\n", imgfile.c_str());

	feature* feat = 0;
	int n = 0; // size of sift features of test image

	n = siftFeature(imgfile.c_str(), &feat, DoubleImg, ContrThr, MaxNkps, CurThr, ContrWeight);
	// word id of keypoints of query image
	int* wids = new int[n];
	// key - word id, value - number of words with word id in the query image
	map<int, int> nterms;
	for (int i=0; i<n; ++i)
	{
		wids[i] = queryKeypoint(feat[i].descr, type);
		nterms[wids[i]]++;
	}

	// tfidfs of keypoints of query image
	float* tfidfs = new float[n];
	// key - fid, value - matched keypoint id pairs, notice the first id is index of features.
	map<int, deque<pair<float, float> > > result;
	// factor of query image
	//float factor = 0.0f;
	for (int i=0; i<n; ++i)
	{
		int ni = wids[i];
		tfidfs[i] = (nterms[ni] * 0.5f / n + 0.5f) * NonLeaves[ni].idf;
		//factor += tfidfs[i] * tfidfs[i];

		for (int j=0; j<NonLeaves[ni].sons.size(); ++j)
		{
			int id = NonLeaves[ni].sons[j].first;
			KeyPoint& kp = KeyPoints[id];
			if (result.find(kp.fid) == result.end())
			{
				result.insert(pair<int, deque<pair<float, float> >>(kp.fid, deque<pair<float, float> >()));
			}
			result[kp.fid].push_back(pair<float, float>(tfidfs[i], kp.tfidf));
		}
	}
	//factor = sqrt((double)factor);

	// key - cosine similarity, value - file id of matched file
	priority_queue<pair<float, int>, vector<pair<float, int> >, greater<pair<float, int> > > pq;
	for (map<int, deque<pair<float, float> > >::const_iterator it=result.begin(); it!=result.end(); ++it)
	{
		float cosine = 0.0f;
		for (int i=0; i<it->second.size(); ++i)
		{
			cosine += it->second[i].first * it->second[i].second;
		}
		cosine /= Factors[it->first]; // no need to divide with the factor of query image

		pq.push(pair<float, int>(cosine, it->first));
		if (pq.size() > TopK)
		{
			pq.pop();
		}
	}

	string qt = getObjectTypeFromPath(imgfile);
	int correct = 0;
	int wrong = 0;
	float score = 0.0f;
	while (!pq.empty())
	{
		pair<float, int> p = pq.top();
		string rt = getObjectTypeFromPath(Fnames[p.second]);
		if (qt == rt)
		{
			++correct;
			score += Wis[(pq.size() - 1) / 5];
		}
		else
		{
			++wrong;
		}

		pq.pop();
	}
	score /= SumWi;

	string category = getObjectCategoryFromPath(imgfile);
	if (Scores.find(category) == Scores.end())
	{
		Scores[category] = Score();
	}
	Scores[category].add(correct, wrong, score);

	Log << boost::format("%s %d %d %f") % imgfile.c_str() % correct % wrong % score << endl;

	delete[] wids;
	delete[] tfidfs;

}

void queryLab()
{
	for (int i=0; i<QueryImgs1.size(); ++i)
	{
		queryOneImg(QueryImgs1[i], 1);
	}

	for (int i=0; i<QueryImgs2.size(); ++i)
	{
		queryOneImg(QueryImgs2[i], 2);
	}

	for (map<string, Score>::const_iterator it=Scores.begin(); it!=Scores.end(); ++it)
	{
		int correct = it->second.correct;
		int wrong = it->second.wrong;
		float score = it->second.score();
		Result << boost::format("%s %d %d %f") % it->first % correct % wrong % score << endl;
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
		("wordfile,w", po::value<string>(&WordsFile), "sift visual words index file containing TF/IDF weight.")
		("resultFile,r", po::value<string>(&ResultFile), "result directory.")
		("logfile,l", po::value<string>(&LogFile), "log file.")
		("topk,k", po::value<int>(&TopK)->default_value(20), "top k results")
		("maxn,n", po::value<int>(&MaxNkps)->default_value(600), "max n")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("queryimgs") == 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0
		|| vm.count("wordfile") == 0 || vm.count("resultFile") == 0 || vm.count("logfile") == 0)
	{
		cout << desc;
		return 1;
	}

	Log.open(LogFile.c_str(), ios_base::app);
	Result.open(ResultFile.c_str());
	Log << "filename correct wrong score" << endl;
	Result << "category correct wrong score" << endl;

	initQuery();

	printf("begin query lab...\n");
	queryLab();

	releaseMemory();

	Log.close();
	Result.close();
	return 0;
}
