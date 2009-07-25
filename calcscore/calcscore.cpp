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

string ImgsFile;
string ResultFile;
string ReDetailFile;

// key - image file id, value - image file name
map<int, string> ImgNames;
// first - image file id, second - image more similarity first, (first-image id, second-similarity)
deque<pair<int, deque<pair<int, float> > > > ImgOrders;

// scores of each categories
int NScore;

char buf[256];
char bigBuf[10240];

class Score
{
public:
	vector<int> correct;
	vector<int> wrong;
	vector<float> sumScore;
	int count;
public:
	Score() 
	{
		this->correct.resize(NScore, 0);
		this->wrong.resize(NScore, 0);
		this->sumScore.resize(NScore, 0.0f);
		this->count = 0;
	}

	Score(int n) 
	{
		this->correct.resize(n, 0);
		this->wrong.resize(n, 0);
		this->sumScore.resize(n, 0.0f);
		this->count = 0;
	}

	Score(const Score& s)
	{
		this->correct = s.correct;
		this->wrong = s.wrong;
		this->sumScore = s.sumScore;
		this->count = s.count;
	}

	void add(const Score& s)
	{
		add(s.correct, s.wrong, s.sumScore);
	}

	void add(const vector<int>& correct, const vector<int>& wrong, const vector<float>& score)
	{
		int n = this->correct.size();
		for (int i=0; i<n; ++i)
		{
			this->correct[i] += correct[i];
			this->wrong[i] += wrong[i];
			this->sumScore[i] += score[i];
		}

		++count;
	}

	// get average score of the category type
	float score(int i) const{ return (count == 0) ? 0 : sumScore[i] / count; }
};

class ScoreComp20
{
public:
	bool operator()(const Score& s1, const Score& s2)
	{
		return s1.sumScore[1] > s2.sumScore[1];
	}
};

// key-category type, value-(key-query image id, value-score of image, bigger score first)
map<string, multimap<Score, int, ScoreComp20> > ImgScores;
// result scores, string-category type, score-score of category, for best 10, 20, 30, 40 images in each category.
map<string, Score> CateScores;;
map<string, Score> CateScores10;


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

void initQuery()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	if (fimg == NULL)
	{
		printf("cannot open %s.\n", ImgsFile.c_str());
		return;
	}
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		ImgNames[fid] = string(buf);
	}
	fclose(fimg);
	cout << "load image list finished." << endl;


	FILE* fdet = fopen(ReDetailFile.c_str(), "r");
	if (fdet == NULL)
	{
		printf("cannot open %s.\n", ReDetailFile.c_str());
		return;
	}

	int fid2;
	int m;
	float sim;
	int count = 0;
	while (fscanf(fdet, "%d", &fid) != EOF)
	{
		fscanf(fdet, "%d", &m);
		ImgOrders.push_back(pair<int, deque<pair<int, float> > >(fid, deque<pair<int, float> > ()));
		for (int i=0; i<m; ++i)
		{
			fscanf(fdet, "%d %f", &fid2, &sim);
			ImgOrders[count].second.push_front(pair<int, float>(fid2, sim));
		}
		++count;
	}

	fclose(fdet);
	cout << "load detail file finished." << endl;

	//NScore = ImgOrders[0].second.size() / 10;
}

const int ScoreL = 4;
float Wis[ScoreL] = {2.0, 1.5, 1.0, 0.5};
const float SumWi = 2.0 + 1.5 + 1.0 + 0.5;

// order - begin from 1, k - topk.
float calcScore(int order, int k)
{
	int t = k / ScoreL;
	int i = (order - 1) / t;
	return Wis[i >= ScoreL ? ScoreL-1 : i];
}

float calcSumWi(int k)
{
	int t = k / ScoreL;
	return SumWi * t + Wis[ScoreL-1] * (k - t * ScoreL);
}

void calcAllScores()
{
	// calculate score of images
	int n = ImgOrders.size();
	for (int i=0; i<n; ++i)
	{
		string qt = getObjectCategoryFromPath(ImgNames[ImgOrders[i].first]);

		Score score(NScore);
		deque<pair<int, float> >& matched = ImgOrders[i].second;
		for (int j=0; j<matched.size(); ++j)
		{
			string rt = getObjectCategoryFromPath(ImgNames[matched[j].first]);
			int order = j + 1;
			if (qt == rt)
			{
				// calculate a list of scores from 10, 20, ..., TopK
				for (int i=0; i<NScore; ++i)
				{
					int k = (i + 1) * 10;
					if (order <= k)
					{
						++score.correct[i];
						score.sumScore[i] += calcScore(order, k);
					}
				}
			}
			else
			{
				for (int i=0; i<NScore; ++i)
				{
					int k = (i + 1) * 10;
					if (order <= k)
					{
						++score.wrong[i];
					}
				}
			}
		}

		for (int i=0; i<NScore; ++i)
		{
			int k = (i + 1) * 10;
			float sw = calcSumWi(k);
			score.sumScore[i] /= sw;
		}
		score.count = 1;

		ImgScores[qt].insert(pair<Score, int>(score, ImgOrders[i].first));
	}

	// calculate score of category.
	for (map<string, multimap<Score, int, ScoreComp20> >::const_iterator it=ImgScores.begin(); it!=ImgScores.end(); ++it)
	{
		CateScores.insert(pair<string, Score>(it->first, Score(NScore)));
		CateScores10.insert(pair<string, Score>(it->first, Score(NScore)));
		int c = 0;
		for (multimap<Score, int, ScoreComp20>::const_iterator it2=it->second.begin(); it2!=it->second.end(); ++it2)
		{
			CateScores[it->first].add(it2->first);
			if (c++ < 10)
			{
				CateScores10[it->first].add(it2->first);
			}
		}
	}
}

void printOneCateScore(const map<string, Score>& cateScore, const string& fname)
{
	ofstream outAll(fname.c_str());
	outAll << "category";
	for (int i=0; i<NScore; ++i)
	{
		int k = (i + 1) * 10;
		outAll << boost::format(" correct%d wrong%d score%d") % k % k % k;
	}
	outAll << endl;

	for (map<string, Score>::const_iterator it=cateScore.begin(); it!=cateScore.end(); ++it)
	{
		outAll << it->first;

		const Score& s = it->second;
		for (int i=0; i<NScore; ++i)
		{
			outAll << boost::format(" %d %d %f") % s.correct[i] % s.wrong[i] % s.score(i);
		}
		outAll << endl;
	}
	outAll.close();
}

void saveCateScores()
{
	printOneCateScore(CateScores, ResultFile + "_all.txt");
	printOneCateScore(CateScores10, ResultFile + "_n10.txt");
}

int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("resultDetail,v", po::value<string>(&ReDetailFile), "input result detail: matched image ids of each query images.")
		("resultFile,r", po::value<string>(&ResultFile), "prefix of output result file.")
		("nscores,n", po::value<int>(&NScore)->default_value(5), "number of scores of each category.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("imgslist") == 0 || vm.count("resultDetail") == 0
		|| vm.count("resultFile") == 0)
	{
		cout << desc;
		return 1;
	}

	initQuery();

	printf("begin calculate scores...\n");
	calcAllScores();

	printf("begin save to file...\n");
	saveCateScores();

	return 0;
}