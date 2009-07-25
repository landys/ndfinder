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
#include <numeric>
#include <utility>
#include <functional>
#include <ctime>
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
	float tfidf;
	int cid; // cluster id, each keypoint belongs to only one cluster
	vector<int> segs;
	KeyPoint() : fid(0), x(0.0f), y(0.0f), scl(0.0f), ori(0.0f), tfidf(0.0f) {}
};

string InfoFile;
string ImgsFile;
string WordsFile;
//string ResultDir;
//string ResultFile;
//string LogFile;
string ReDetailFile;
string QueryImgsFile;
string SegFile;

//ofstream Log;
//ofstream Result;
ofstream Detail;

class ImageInfo
{
public:
	string fname; // image file name
	float factor; // standard factors for tf/idf
	vector<int> kps; // keypoint ids
	ImageInfo() : factor(0.0f) {}
};

/*class MatchPair
{
public:
	// first-seg id of a file
	int segId1;
	int segId2;
	// keypoints matched between the two regions, so the size of kps1 and kps2 is the same.
	vector<int> kps1;
	vector<int> kps2;
};*/

// all keypoints in dataset, index is keypoint id, value is file id.
deque<KeyPoint> KeyPoints;
// key - image file id, value - image information
map<int, ImageInfo> ImgInfos;
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
// query image list read from file, first - image id, second - image file name
deque<pair<int, string> > QueryImgs;
// partition id
//int PartId;
// top k query
int TopK;
// Lambda of the similarity
float Lambda;
// Piu of the similarity
float Piu;
// limit of matched segment regions
int MatchLimit;

//class Score
//{
//public:
//	int correct;
//	int wrong;
//	float sumScore;
//	long sumSiftTime; // ms
//	long sumTotalTime; // ms
//
//	Score() : correct(0), wrong(0), sumScore(0.0f), sumSiftTime(0), sumTotalTime(0) {}
//
//	void add(int correct, int wrong, float score, long siftTime, long totalTime)
//	{
//		this->correct += correct;
//		this->wrong += wrong;
//		this->sumScore += score;
//		this->sumSiftTime += siftTime;
//		this->sumTotalTime += totalTime;
//	}
//
//	// get average score of the category type
//	float score() const{ return (correct + wrong == 0) ? 0 : sumScore * TopK / (correct + wrong); }
//	long siftTime() const { return (correct + wrong == 0) ? 0 : sumSiftTime * TopK / (correct + wrong); }
//	long totalTime() const { return (correct + wrong == 0) ? 0 : sumTotalTime * TopK / (correct + wrong); }
//};
//// result scores, key-category type, value-score of category
//map<string, Score> Scores;
//
//int Interval = 20;
//
//float Wis[4] = {2.0, 1.5, 1.0, 0.5};
//float SumWi = (2.0 + 1.5 + 1.0 + 0.5) * 5;
//
//int DoubleImg = 1;
//double ContrThr = 0.03;
//int MaxNkps = 2500;
//double CurThr = 10;
//double ContrWeight = 0.6;

char buf[256];
char bigBuf[10240];


string& findPicByKpId(int key)
{
	return ImgInfos[KeyPoints[key].fid].fname;
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
	return getObjectCategoryFromPath(fn);
	//int i = fn.rfind('/');
	//if (i == -1)
	//{
	//	i = fn.rfind('\\');
	//}

	//int j = fn.rfind('/', i-1);
	//if (j == -1)
	//{
	//	j = fn.rfind('\\', i-1);
	//}

	//string re = fn.substr(j+1, i-j-1);

	//i = re.find('_');
	//if (i != -1)
	//{
	//	re = re.substr(0, i);
	//}

	//return re;
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
	int begLeaf = NNodes - RealNNodes;
	for (int i=0; i<NNodes; ++i)
	{
		NonLeaves.push_back(Node());
		fscanf(fw, "%d %d %f %d", &id, &NonLeaves[i].npoints, &NonLeaves[i].idf, &ns);
		NonLeaves[i].sons.resize(ns);
		if (i >= begLeaf)
		{
			for (int j=0; j<ns; ++j)
			{
				fscanf(fw, "%d %f", &NonLeaves[i].sons[j].first, &NonLeaves[i].sons[j].second);
				// get tfidf
				KeyPoints[NonLeaves[i].sons[j].first].tfidf = NonLeaves[i].sons[j].second * NonLeaves[i].idf;
				// assign cluster id
				KeyPoints[NonLeaves[i].sons[j].first].cid = i;
			}
		}
		else
		{
			for (int j=0; j<ns; ++j)
			{
				fscanf(fw, "%d %f", &NonLeaves[i].sons[j].first, &NonLeaves[i].sons[j].second);
			}
		}

		NonLeaves[i].center = new float[Dim];
		for (int j=0; j<Dim; ++j)
		{
			fscanf(fw, "%f", &NonLeaves[i].center[j]);
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
		ImgInfos[KeyPoints[i].fid].factor += KeyPoints[i].tfidf * KeyPoints[i].tfidf;
	}

	for (map<int, ImageInfo>::iterator it=ImgInfos.begin(); it!=ImgInfos.end(); ++it)
	{
		it->second.factor = sqrt((double)it->second.factor);
	}

	cout << "calculate factors of images finished: " <<  ImgInfos.size() << endl;
}

void initQuery()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		ImgInfos[fid].fname = string(buf);
	}
	fclose(fimg);
	cout << "load image list finished." << endl;

	FILE* fids = fopen(InfoFile.c_str(), "r");
	KeyPoint kp;
	int id;
	int count = 0;
	while (fscanf(fids, "%d %d %f %f %f %f", &id, &kp.fid, &kp.x, &kp.y, &kp.scl, &kp.ori) != EOF)
	{
		KeyPoints.push_back(kp);
		ImgInfos[kp.fid].kps.push_back(count++);
		// NOTICE if one line of the id file is more than 1024, error will happen.
		fgets(bigBuf, 10240, fids);
	}

	fclose(fids);
	cout << "load file info finished." << endl;

	FILE* fseg = fopen(SegFile.c_str(), "r");
	if (fseg == 0)
	{
		printf("Cannot open %s for read.\n", SegFile.c_str());
		return ;
	}
	int nseg, segid;
	count = 0;
	while (fscanf(fseg, "%d %d %d", &id, &fid, &nseg) != EOF)
	{
		for (int i=0; i<nseg; ++i)
		{
			fscanf(fseg, "%d", &segid);
			KeyPoints[count].segs.push_back(segid);
		}
		++count;
	}
	fclose(fseg);
	cout << "load segmentation info finished." << endl;

	// read tfidf word file
	readWordsFile();

	// calculate standard factor for images
	calFactors();

	// read query image file
	FILE* fq = fopen(QueryImgsFile.c_str(), "r");
	while (fscanf(fimg, "%d %s", &id, buf) != EOF)
	{
		QueryImgs.push_back(pair<int, string>(id, string(buf)));
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

// calculate difference value of orientations of p and q, which is q-p accually.
float calcOriDiff(const vector<int>& p, const vector<int>& q)
{
	int n = min(p.size(), q.size());

	float sumd = 0.0f;
	float sumscl = 0.0f;
	for (int i=0; i<n; ++i)
	{
		float ascl = (KeyPoints[p[i]].scl + KeyPoints[q[i]].scl) / 2;
		float dori = KeyPoints[q[i]].ori - KeyPoints[p[i]].ori;
		sumd += ascl * dori;
		sumscl += ascl;
	}

	return sumd / sumscl;
}

// sort for Mg
class GeoSerial
{
public:
	float x;
	float y;
	int ser;
	GeoSerial() : x(0.0f), y(0.0f), ser(0) {}
	GeoSerial(float x, float y, int ser) : x(x), y(y), ser(ser) {}
};

class CompGsX
{
public:
	bool operator() (const GeoSerial& gs1, const GeoSerial& gs2)
	{
		return gs1.x < gs2.x;
	}
};

class CompGsY
{
public:
	bool operator() (const GeoSerial& gs1, const GeoSerial& gs2)
	{
		return gs1.y < gs2.y;
	}
};


// dori = oriq - orip
int calcMg(const vector<int>& p, const vector<int>& q, float dori)
{
	int n = min(p.size(), q.size());
	if (n <= 1)
	{
		return 0.0f;
	}

	// rotate p by angle dori
	float dcos = cos((double)dori);
	float dsin = sin((double)dori);
	vector<GeoSerial> gp(n);
	vector<GeoSerial> gq(n);
	for (int i=0; i<n; ++i)
	{
		gp[i].x = KeyPoints[p[i]].x * dcos - KeyPoints[p[i]].y * dsin;
		gp[i].y = KeyPoints[p[i]].x * dsin + KeyPoints[p[i]].y * dcos;
		gp[i].ser = i;
	}

	// calc mgx
	sort(gp.begin(), gp.end(), CompGsX());
	for (int i=0; i<n; ++i)
	{
		gq[gp[i].ser].x = KeyPoints[q[i]].x;
		gq[gp[i].ser].y = KeyPoints[q[i]].y;
		gq[i].ser = i;
	}
	sort(gq.begin(), gq.end(), CompGsX());
	int mgx = 0;
	for (int i=0; i<n-1; ++i)
	{
		if (gq[i+1].ser < gq[i].ser)
		{
			mgx -= 1;
		}
	}

	// calc mgy
	sort(gp.begin(), gp.end(), CompGsY());
	for (int i=0; i<n; ++i)
	{
		gq[gp[i].ser].x = KeyPoints[q[i]].x;
		gq[gp[i].ser].y = KeyPoints[q[i]].y;
		gq[i].ser = i;
	}
	sort(gq.begin(), gq.end(), CompGsY());
	int mgy = 0;
	for (int i=0; i<n-1; ++i)
	{
		if (gq[i+1].ser < gq[i].ser)
		{
			mgy -= 1;
		}
	}

	return min(mgx, mgy);
}

//void gauss_normalization(float* arr, int n)
//{
//	if (n == 0)
//	{
//		return;
//	}
//
//	int i = 0;
//	float total = 0.0;
//	for (i=0; i<n; ++i)
//	{
//		total += arr[i];
//	}
//	float avr = total / n;
//
//	total = 0.0;
//	for (i=0; i<n; ++i)
//	{
//		total += (arr[i] - avr) * (arr[i] - avr);
//	}
//	float len = total != 0 ? 1.0 / sqrt(total) : 1.0;
//
//	for (i=0; i<n; ++i)
//	{
//		arr[i] = ((arr[i] - avr) * len + 1.0) / 2;
//	}
//}

const double Pi = 3.14159265;
float calcMd(const vector<int>& p, const vector<int>& q, float dori)
{
	int n = min(p.size(), q.size());
	if (n <= 1)
	{
		return 0.0f;
	}

	// normalize "/ (4 * Pi) + 0.5" to [0, 1].
	float* oris = new float[n];
	for (int i=0; i<n; ++i)
	{
		oris[i] = (KeyPoints[q[i]].ori - KeyPoints[p[i]].ori) / (4 * Pi) + 0.5;
	}
	//gauss_normalization(oris, n);

	//float sum = 0.0f;
	//for (int i=0; i<n; ++i)
	//{
	//	sum += oris[i];
	//}
	//float aver = sum / n;

	float aver = dori / (4 * Pi) + 0.5;
	float sum = 0.0f;
	for (int i=0; i<n; ++i)
	{
		sum += (oris[i] - aver) * (oris[i] - aver);
	}

	delete[] oris;
	return -sqrt((double)(sum / n));
}


void queryOneImg(int imgId, const string& imgfile)
{
	printf("Query %s...\n", imgfile.c_str());

	long time1 = clock();
	//feature* feat = 0;
	//int n = 0; // size of sift features of test image

	//n = siftFeature(imgfile.c_str(), &feat, DoubleImg, ContrThr, MaxNkps, CurThr, ContrWeight);
	long time2 = clock();

	// word id of keypoints of query image
	//int* wids = new int[n];
	//// key - word id, value - number of words with word id in the query image
	//map<int, int> nterms;
	//for (int i=0; i<n; ++i)
	//{
	//	wids[i] = queryKeypoint(feat[i].descr, type);
	//	nterms[wids[i]]++;
	//}

	//// tfidfs of keypoints of query image
	//float* tfidfs = new float[n];
	// key - fid, value - matched keypoint id pairs, notice the first id is index of features.
	//map<int, deque<pair<float, float> > > result;

	// key-fid, value-(key-(first-segId of query image, second-segId of matched image), 
	// value-(first-matched keypoints of seg of query image, second-matched keypoints of seg of matched image))
	map<int, map<pair<int, int>, pair<vector<int>, vector<int> > > > result;
	// factor of query image
	//float factor = 0.0f;
	vector<int>& kps = ImgInfos[imgId].kps;
	// the times keypoints is the matched keypoints
	//map<int, int> kpTimes;
	int n = kps.size();
	for (int i=0; i<n; ++i)
	{
		//int ni = wids[i];
		//tfidfs[i] = (nterms[ni] * 0.5f / n + 0.5f) * NonLeaves[ni].idf;
		//factor += tfidfs[i] * tfidfs[i];

		KeyPoint& qk = KeyPoints[kps[i]];
		if (qk.segs.size() == 0)
		{
			// the keypoint belongs no segs, so ignored
			continue;
		}

		int ni = qk.cid;
		for (int j=0; j<NonLeaves[ni].sons.size(); ++j)
		{
			int id = NonLeaves[ni].sons[j].first;
			KeyPoint& kp = KeyPoints[id];
			// add the segment number of its matched keypoint. no use after test
			/*kpTimes[kps[i]] += kp.segs.size();
			kpTimes[id] += qk.segs.size();*/

			for (int si=0; si<kp.segs.size(); ++si)
			{
				for (int sj=0; sj<qk.segs.size(); ++sj)
				{
					pair<vector<int>, vector<int> >& rp = result[kp.fid][pair<int, int>(qk.segs[sj], kp.segs[si])];
					rp.first.push_back(kps[i]);
					rp.second.push_back(id);
					// update times of matched keypoints.
				}
			}
			//if (result.find(kp.fid) == result.end())
			//{
			//	result.insert(pair<int, deque<pair<float, float> >>(kp.fid, deque<pair<float, float> >()));
			//}
			////result[kp.fid].push_back(pair<float, float>(tfidfs[i], kp.tfidf));
			//result[kp.fid].push_back(pair<float, float>(KeyPoints[kps[i]].tfidf, kp.tfidf));
		}
	}
	//factor = sqrt((double)factor);

	// key - similarity value, value - file id of matched file. small value first in priority_queue.
	// for similarity value, bigger value means more similar
	priority_queue<pair<float, int>, vector<pair<float, int> >, greater<pair<float, int> > > pq;
	for (map<int, map<pair<int, int>, pair<vector<int>, vector<int> > > >::const_iterator it=result.begin(); it!=result.end(); ++it)
	{
		float cosine = 0.0f;
		for (map<pair<int, int>, pair<vector<int>, vector<int> > >::const_iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
		{
			if (it2->second.first.size() < MatchLimit)
			{
				continue;
			}
			const vector<int>& vp = it2->second.first;
			const vector<int>& vq = it2->second.second;
			float dori = calcOriDiff(vp, vq);
			int mm = vp.size();
			int mg = calcMg(vp, vq, dori);
			float md = calcMd(vp, vq, dori);

			float mall = mm + Lambda * mg + Piu * mm * md;
			// mall < 0 is meanless
			if (mall < 0.0f)
			{
				mall = 0.0f;
			}

			for (int i=0; i<mm; ++i)
			{
				cosine += mall * KeyPoints[vp[i]].tfidf * KeyPoints[vq[i]].tfidf / mm; // (KeyPoints[vp[i]].segs.size() * KeyPoints[vq[i]].segs.size());// / (kpTimes[vp[i]] * kpTimes[vq[i]]);
			}
		}

		//float cosine = 0.0f;
		//for (int i=0; i<it->second.size(); ++i)
		//{
		//	cosine += it->second[i].first * it->second[i].second;
		//}
		cosine /= ImgInfos[it->first].factor; // no need to divide with the factor of query image

		pq.push(pair<float, int>(cosine, it->first));
		if (pq.size() > TopK)
		{
			pq.pop();
		}
	}
	long time3 = clock();

	Detail << boost::format("%d %d") % imgId % pq.size();

	// calculate scores
	//string qt = getObjectTypeFromPath(imgfile);
	//int correct = 0;
	//int wrong = 0;
	//float score = 0.0f;
	while (!pq.empty())
	{
		pair<float, int> p = pq.top();
		Detail << boost::format(" %d %f") % p.second % p.first;
		//string rt = getObjectTypeFromPath(ImgInfos[p.second].fname);
		//if (qt == rt)
		//{
		//	++correct;
		//	score += Wis[(pq.size() - 1) / 5];
		//}
		//else
		//{
		//	++wrong;
		//}

		pq.pop();
	}
	Detail << endl;
	//score /= SumWi;

	//string category = getObjectCategoryFromPath(imgfile);
	//if (Scores.find(category) == Scores.end())
	//{
	//	Scores[category] = Score();
	//}
	//Scores[category].add(correct, wrong, score, time2-time1, time3-time1);


	//Log << boost::format("%s %d %d %f %ld %ld %ld") % imgfile.c_str() % correct % wrong % score % (time3-time1) % (time2-time1) % (time3-time2) << endl;

	//delete[] wids;
	//delete[] tfidfs;
	//free(feat);
}

void queryLab()
{
	for (int i=0; i<QueryImgs.size(); ++i)
	{
		queryOneImg(QueryImgs[i].first, QueryImgs[i].second);
	}

	//for (map<string, Score>::const_iterator it=Scores.begin(); it!=Scores.end(); ++it)
	//{
	//	int correct = it->second.correct;
	//	int wrong = it->second.wrong;
	//	float score = it->second.score();
	//	long siftTime = it->second.siftTime();
	//	long totalTime = it->second.totalTime();
	//	Result << boost::format("%s %d %d %f %ld %ld %ld") % it->first % correct % wrong % score % totalTime % siftTime % (totalTime-siftTime) << endl;
	//}
}


int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("queryimgs,q", po::value<string>(&QueryImgsFile), "the query image list file with ids.")
		("datamap,m", po::value<string>(&InfoFile), "the map file between pictures and their keypoints.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("wordfile,w", po::value<string>(&WordsFile), "sift visual words index file containing TF/IDF weight.")
		("segfile,s", po::value<string>(&SegFile), "segmentation info file.")
		//("resultFile,r", po::value<string>(&ResultFile), "result directory.")
		("resultDetail,v", po::value<string>(&ReDetailFile), "result detail: matched image ids of each query images.")
		//("logfile,l", po::value<string>(&LogFile), "log file.")
		("lambda,d", po::value<float>(&Lambda)->default_value(2.0f), "lamdda of similarity expression.")
		("Piu,p", po::value<float>(&Piu)->default_value(1.0f), "Piu of similarity expression.")
		("matchlimit,t", po::value<int>(&MatchLimit)->default_value(2), "limit of keypoints of matched regions.")
		("topk,k", po::value<int>(&TopK)->default_value(50), "top k results")
		//("maxn,n", po::value<int>(&MaxNkps)->default_value(600), "max n")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("queryimgs") == 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0
		|| vm.count("wordfile") == 0 || vm.count("segfile") == 0 || vm.count("resultDetail") == 0)
	{
		cout << desc;
		return 1;
	}

	//Log.open(LogFile.c_str(), ios_base::app);
	//Result.open(ResultFile.c_str());
	//Log << "filename correct wrong score total_time sift_time query_time" << endl;
	//Result << "category correct wrong score total_time sift_time query_time" << endl;
	Detail.open(ReDetailFile.c_str());

	initQuery();

	printf("begin query lab...\n");
	queryLab();

	releaseMemory();

	//Log.close();
	//Result.close();
	Detail.close();
	return 0;
}
