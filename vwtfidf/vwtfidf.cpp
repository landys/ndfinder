#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <deque>
#include <algorithm>
#include <queue>
#include <numeric>
#include <utility>
#include <functional>
using namespace std;
namespace po = boost::program_options;


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
};

string InfoFile;
string ImgsFile;
string WordsFile;
string ResultFile;
string LogFile;

ofstream Log;

// all keypoints in dataset, index is keypoint id, value is file id.
deque<KeyPoint> KeyPoints;
// key - file id, value - file name
map<int, string> Fnames;
// key - file id, value - keypoints
map<int, int> Fkps;
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


char buf[256];
char bigBuf[10240];


string& findPicByKpId(int key)
{
	return Fnames[KeyPoints[key].fid];
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
		NonLeaves[ni].sons.resize(sons);
		for (int j=0; j<sons; ++j)
		{
			fscanf(fw, "%d", &NonLeaves[ni].sons[j].first);
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

void printTfIdfWordsFile()
{
	FILE* fw = fopen(ResultFile.c_str(), "w");
	if (fw == 0)
	{
		printf("Cannot open %s\n", ResultFile.c_str());
		return;
	}

	fprintf(fw, "%d %d %d %d\n", N, NonLeaves.size(), RealNNodes, Dim);
	for (int i=0; i<NonLeaves.size(); ++i)
	{
		fprintf(fw, "%d %d %g %d ", i, NonLeaves[i].npoints, NonLeaves[i].idf, NonLeaves[i].sons.size());
		for (int j=0; j<NonLeaves[i].sons.size(); ++j)
		{
			fprintf(fw, "%d %g ", NonLeaves[i].sons[j].first, NonLeaves[i].sons[j].second);
		}
		for (int j=0; j<Dim; ++j)
		{
			fprintf(fw, "%g ", NonLeaves[i].center[j]);
		}
		fprintf(fw, "\n");
	}

	fclose(fw);
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
		Fkps[kp.fid]++;
		// NOTICE if one line of the id file is more than 1024, error will happen.
		fgets(bigBuf, 10240, fids);
	}

	fclose(fids);
	cout << "load file info finished." << endl;

	readWordsFile();
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


void genTfIdf()
{
	int begLeaf = NNodes - RealNNodes;
	for (int i=begLeaf; i<NNodes; ++i)
	{
		vector<pair<int, float> >& sons = NonLeaves[i].sons;
		int ns = sons.size();
		// key - file id, value - number of this word in the file
		map<int, int> fn;
		int ni = ns;
		for (int j=0; j<ns; ++j)
		{
			int fid = KeyPoints[sons[j].first].fid;
			if (fn.find(fid) != fn.end())
			{
				ni--;
			}
			fn[fid]++;
		}

		float idf = log((double)N / ni);
		NonLeaves[i].idf = idf;
		for (int j=0; j<ns; ++j)
		{
			int fid = KeyPoints[sons[j].first].fid;
			sons[j].second = (float)fn[fid] / Fkps[fid]; // only tf
		}
	}

	printTfIdfWordsFile();
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
		("resultFile,r", po::value<string>(&ResultFile), "result directory.")
		//("logfile,l", po::value<string>(&LogFile), "log file.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0
		|| vm.count("wordfile") == 0 || vm.count("resultFile") == 0)
	{
		cout << desc;
		return 1;
	}

	//Log.open(LogFile.c_str(), ios_base::app);

	initQuery();

	printf("begin generate tf/idf...\n");
	genTfIdf();

	releaseMemory();

	//Log.close();
	return 0;
}
